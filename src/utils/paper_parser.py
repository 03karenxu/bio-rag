from __future__ import annotations
 
import re
import tiktoken
import logging
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

from utils.schemas import Paper, Chunk, Reference
from utils.image_processing import estimate_image_tokens, get_image_paths
from config import MIN_CHUNK_TOKENS, COHERE_TRANSFORMABLE_FORMATS, COHERE_COMPATIBLE_FORMATS

logger = logging.getLogger()

class MissingContentError(ValueError):
    pass

_MEDIA_L = "[[["
_MEDIA_R = "]]]"
MEDIA_MARKER = re.compile(re.escape(_MEDIA_L) + r'(.+?)' + re.escape(_MEDIA_R))

_XLINK = "{http://www.w3.org/1999/xlink}href"
_MEDIA_TAGS = {"graphic", "media", "inline-graphic"}
_TITLE_TAGS = {"title", "label"}
_TABLE_MARKER = "table-wrap"

@dataclass
class PaperContext:
    root: ET.Element
    media_folder: Optional[Path] = None

class PaperParser:
    '''
    parses a preprint xml file from biorxiv into a Paper model
    '''

    def __init__(self, token_enc_type: str = "cl100k_base"):
        self.TT = tiktoken.get_encoding(token_enc_type)

    def parse_paper(self, xml: Path, media_folder: Path | None = None) -> Paper:
        root = ET.parse(xml).getroot()
        if root.tag == "pmc-articleset":
            article = root.find("article")
            if article is None:
                raise MissingContentError("pmc-articleset contains no article element")
            root = article
        ctx = PaperContext(root=root, media_folder=media_folder)

        return Paper(
            title=self.get_title(ctx),
            doi=self.get_doi(ctx),
            abstract=self.get_abstract(ctx),
            keywords=self.get_keywords(ctx),
            authors=self.get_authors(ctx),
            date=self.get_date("accepted", ctx) or self.get_date("received", ctx) or self.get_date("published", ctx),
            categories=self.get_categories(ctx),
            body=self.get_body(ctx),
            references=self.get_references(ctx),
        )

    def get_title(self, ctx: PaperContext) -> str:
        e = ctx.root.find("front/article-meta/title-group/article-title") or ctx.root.find(".//article-title")
        return self._get_all_text_with_media(e, ctx.media_folder)

    def get_doi(self, ctx: PaperContext) -> str:
        e = (
            ctx.root.find("./front/article-meta/article-id[@pub-id-type='doi']") or
            ctx.root.find(".//article-id[@pub-id-type='doi']")
        )
        return self._get_all_text_with_media(e, ctx.media_folder)

    def get_abstract(self, ctx: PaperContext) -> list[Chunk]:
        abstract = ctx.root.find(".//abstract")
        if abstract is None:
            raise MissingContentError("no abstract")
        return self._merge_small_chunks(self._process_section(abstract, ctx))

    def get_categories(self, ctx: PaperContext) -> list[str]:
        return [s.text for s in ctx.root.findall(".//subj-group/subject") if s.text]

    def get_keywords(self, ctx: PaperContext) -> list[str]:
        return [t for kwd in ctx.root.findall(".//kwd") if (t := self._get_all_text_with_media(kwd, ctx.media_folder))]

    def get_authors(self, ctx: PaperContext) -> list[str]:
        return [
            f"{self._get_all_text_with_media(c.find('name/surname'), ctx.media_folder)}, {self._get_all_text_with_media(c.find('name/given-names'), ctx.media_folder)}".strip(", ")
            for c in ctx.root.findall(".//contrib[@contrib-type='author']")
        ]

    def get_date(self, date_type: str, ctx: PaperContext) -> date | None:
        if date_type == "published":
            return self._get_pub_date(ctx)

        node = ctx.root.find(f".//history/date[@date-type='{date_type}']")
        if node is None:
            return None
        try:
            return date(
                year=int(node.findtext("year")),
                month=int(node.findtext("month")),
                day=int(node.findtext("day")),
            )
        except (TypeError, ValueError):
            return None

    def get_body(self, ctx: PaperContext) -> list[Chunk]:
        body = ctx.root.find(".//body")
        if body is None:
            raise MissingContentError("no body")
        return self._merge_small_chunks(self._process_section(body, ctx))

    def get_references(self, ctx: PaperContext) -> list[Reference]:
        refs = []
        for ref in ctx.root.findall(".//ref-list/ref"):
            ref_id = ref.get("id", "")
            ec = ref.find(".//element-citation")
            if ec is not None:
                refs.append(self._parse_element_citation(ref_id, ec))
            else:
                mc = ref.find(".//mixed-citation")
                if mc is not None:
                    refs.append(self._parse_mixed_citation(ref_id, mc))
        return refs

    # --------------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        merged = []
        for chunk in chunks:
            chunk = self._remove_incompatible_media(chunk)
            if chunk is None:
                continue
            has_graphic = bool(MEDIA_MARKER.search(chunk.text))
            if chunk.n_tokens < MIN_CHUNK_TOKENS and not has_graphic and merged:
                prev = merged[-1]
                prev.text += " " + chunk.text.strip()
                prev.n_tokens += chunk.n_tokens
            else:
                merged.append(chunk.model_copy())
        return merged

    def _remove_incompatible_media(self, chunk: Chunk) -> Chunk | None:
        markers = list(MEDIA_MARKER.finditer(chunk.text))
        if not markers:
            return chunk
        
        incompatible = [
            m for m in markers
            if Path(m.group(1)).suffix.lower() not in COHERE_TRANSFORMABLE_FORMATS | COHERE_COMPATIBLE_FORMATS
        ]

        if not incompatible:
            return chunk

        # chunk contains cohere-incompatible media

        if chunk.n_tokens < MIN_CHUNK_TOKENS:
            # small chunk containing only incompatible material, discard
            return None
        
        # chunk contains incompatible material but also text, keep text and remove marker
        text = chunk.text
        for m in reversed(incompatible):
            text = text[:m.start()] + text[m.end():]
        text = " ".join(text.split())
        return chunk.model_copy(update={"text": text, "n_tokens": len(self.TT.encode(text))})

    def _get_pub_date(self, ctx: PaperContext) -> date | None:
        for node in ctx.root.findall(".//pub-date"):
            try:
                return date(
                    year=int(node.findtext("year")),
                    month=int(node.findtext("month") or 1),
                    day=int(node.findtext("day") or 1),
                )
            except (TypeError, ValueError):
                continue
        return None
    
    def _process_section(self, sec: ET.Element, ctx: PaperContext, parent_title: str | None = None) -> list[Chunk]:
        title_elem = sec.find("title")
        title = " ".join(self._get_all_text_with_media(title_elem, ctx.media_folder).split()).strip()
        full_title = f"{parent_title} > {title}" if parent_title else title

        chunks = []
        for child in sec:
            if child.tag == "sec":
                chunks.extend(self._process_section(child, ctx, full_title))
                continue
            elif child.tag == _TABLE_MARKER:
                text = self._table_to_markdown(child, ctx.media_folder)
            elif child.tag not in _TITLE_TAGS:
                text = " ".join(self._get_all_text_with_media(child, ctx.media_folder).split())
            else:
                continue
            if text:
                full_text = f"Section: {full_title} Content: {text}"
                n_tokens = len(self.TT.encode(full_text)) + (sum(
                    estimate_image_tokens(path)
                    for m in MEDIA_MARKER.finditer(text)
                    for path in get_image_paths(ctx.media_folder, m.group(1))
                ) if ctx.media_folder else 0)
                chunks.append(Chunk(n_tokens=n_tokens, section=full_title, text=text))
        return chunks

    def _parse_element_citation(self, ref_id: str, ec: ET.Element) -> Reference:
        authors = [
            f"{n.findtext('surname', '')}, {n.findtext('given-names', '')}".strip(", ")
            for n in ec.findall("person-group/name")
        ]
        pages = ""
        fpage, lpage = ec.findtext("fpage", ""), ec.findtext("lpage", "")
        if fpage:
            pages = f"{fpage}–{lpage}" if lpage else fpage
        pub_ids = {p.get("pub-id-type", ""): (p.text or "").strip() for p in ec.findall("pub-id")}
        year_text = ec.findtext("year", "")
        return Reference(
            ref_id=ref_id,
            authors=authors,
            title=re.sub(r'\s+', ' ', ec.findtext("article-title", "")).strip(),
            pub_type=ec.get("publication-type", ""),
            journal=ec.findtext("source", ""),
            year=int(year_text) if year_text and year_text.isdigit() else None,
            volume=ec.findtext("volume", ""),
            pages=pages,
            doi=pub_ids.get("doi", ""),
            pmid=pub_ids.get("pmid", ""),
            pmcid=pub_ids.get("pmcid", ""),
        )

    def _parse_mixed_citation(self, ref_id: str, mc: ET.Element) -> Reference:
        authors = []
        for n in mc.findall(".//name"):
            surname = n.findtext("surname", "")
            given = n.findtext("given-names", "")
            if surname:
                authors.append(f"{surname}, {given}".strip(", "))
        for n in mc.findall(".//string-name"):
            surname = n.findtext("surname", "")
            given = n.findtext("given-names", "")
            if surname:
                authors.append(f"{surname}, {given}".strip(", "))
        for c in mc.findall(".//collab"):
            text = "".join(c.itertext()).strip()
            if text:
                authors.append(text)

        def elem_text(tag: str) -> str:
            e = mc.find(f".//{tag}")
            return "".join(e.itertext()).strip() if e is not None else ""

        title = re.sub(r'\s+', ' ', elem_text("article-title") or elem_text("source") or "").strip()
        journal = elem_text("source") if mc.find(".//article-title") is not None else ""

        pages = ""
        fpage, lpage = mc.findtext("fpage", ""), mc.findtext("lpage", "")
        if fpage:
            pages = f"{fpage}–{lpage}" if lpage else fpage
        pub_ids = {p.get("pub-id-type", ""): (p.text or "").strip() for p in mc.findall(".//pub-id")}
        year_text = mc.findtext(".//year", "")
        raw_text = " ".join("".join(mc.itertext()).split())
        return Reference(
            ref_id=ref_id,
            authors=authors,
            title=title,
            pub_type=mc.get("publication-type", ""),
            journal=journal,
            year=int(year_text) if year_text and year_text.isdigit() else None,
            volume=mc.findtext(".//volume", ""),
            pages=pages,
            doi=pub_ids.get("doi", ""),
            pmid=pub_ids.get("pmid", ""),
            pmcid=pub_ids.get("pmcid", ""),
            raw_text=raw_text,
        )

    def _table_to_markdown(self, table_wrap: ET.Element, media_folder: Path | None = None) -> str:
        label = (table_wrap.findtext("label") or "").strip()
        caption_parts = []
        caption = table_wrap.find("caption")
        if caption is not None:
            for p in caption.iter():
                if p.text and p.text.strip():
                    caption_parts.append(p.text.strip())
        header = f"{label}: {' '.join(caption_parts)}".strip(": ")

        table = table_wrap.find(".//table")
        if table is None:
            return header

        def row_to_cells(tr: ET.Element) -> list[str]:
            return [
                " ".join(self._get_all_text_with_media(cell, media_folder).split())
                for cell in tr
                if cell.tag in ("th", "td")
            ]

        rows: list[list[str]] = []
        thead = table.find("thead")
        if thead is not None:
            for tr in thead.findall("tr"):
                rows.append(row_to_cells(tr))
        tbody = table.find("tbody")
        if tbody is not None:
            for tr in tbody.findall("tr"):
                rows.append(row_to_cells(tr))

        if not rows:
            return header

        # normalize row widths
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]

        col_widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(n_cols)]
        col_widths = [max(w, 1) for w in col_widths]

        def fmt_row(cells: list[str]) -> str:
            return "| " + " | ".join(c.ljust(col_widths[j]) for j, c in enumerate(cells)) + " |"

        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"

        lines = [fmt_row(rows[0]), separator] + [fmt_row(r) for r in rows[1:]]
        table_md = "\n".join(lines)
        return f"{header}\n{table_md}" if header else table_md

    def _get_all_text_with_media(self, e: ET.Element, media_folder: Path) -> str:
        '''
        gets all the text inside an element, formatting graphics as
        <_MEDIA_L>PATH_TO_GRAPHIC<_MEDIA_R> inline.
        '''
        if e is None:
            return ""
        parts: list[str] = []
        if e.text:
            parts.append(e.text)
        for child in e:
            if child.tag in _MEDIA_TAGS and _XLINK in child.attrib:
                img_name = Path(child.attrib[_XLINK]).name
                if media_folder:
                    parts.append(f"{_MEDIA_L}{img_name}{_MEDIA_R}")
            elif child.tag == _TABLE_MARKER:
                parts.append(self._table_to_markdown(child, media_folder))
            else:
                parts.append(self._get_all_text_with_media(child, media_folder))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)


