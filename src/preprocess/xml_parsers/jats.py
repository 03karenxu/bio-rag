from __future__ import annotations

import re
import hashlib
import logging
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters

from preprocess.xml_parsers.schema import *

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
#  public api
# ------------------------------------------------------------------------------

def parse_file(path: Path) -> Paper:
    tree = ET.parse(path)
    root = tree.getroot()
    el = root.find("article") or root
    return _parse_article(el)

def parse_string(xml_text: str) -> Paper:
    root = ET.fromstring(xml_text)
    el = root.find("article") or root
    return _parse_article(el)

# ------------------------------------------------------------------------------

## globals ##

_XLINK = "{http://www.w3.org/1999/xlink}href"
_TITLE_TAGS = {"title", "label"}
_RANGE_DASHES = {"–", "—", "-", "‐", "‑", "−"}

_punkt_params = PunktParameters()
_punkt_params.abbrev_types = {
    "fig", "figs", "et al", "e.g", "i.e", "vs", "c.a", "approx", "dept", "e"
}
_tokenizer = PunktSentenceTokenizer(_punkt_params)

## top-level ##

def _parse_article(article: ET.Element) -> Paper:
    front_el = article.find("front")
    if front_el is None: raise MissingContentError(element="front", e=article)

    body = article.find("body")
    if body is None:raise MissingContentError(element="body", e=article)
    sections, media, tables = _parse_body(body)

    back = article.find("back")
    
    return Paper(
        front=_parse_front(front_el),
        body=sections,
        media=media,
        inline_tables=tables,
        references=_parse_references(back),
    )

## front ##

def _parse_front(front: ET.Element | None) -> Front:
    meta = front.find("article-meta")
    if meta is None: raise MissingContentError(element="meta", e=front)

    # get article title
    title_el    = meta.find(".//article-title")
    title       = _element_text(title_el) if title_el is not None else ""

    authors     = _parse_authors(meta)
    pub_date    = _parse_date(meta)
    abstract =   _parse_abstract(meta)

    return Front(
        title=title,
        authors=authors,
        hash=hashlib.md5(title.encode()).hexdigest().upper(),
        publication_date=pub_date.isoformat() if pub_date else "",
        abstract=abstract,
    )

def _parse_authors(meta: ET.Element) -> list[str]:
    authors = []
    for contrib in meta.findall(".//contrib[@contrib-type='author']"):
        name_el = contrib.find("name")
        if name_el is not None:
            surname = name_el.findtext("surname", "").strip()
            given   = name_el.findtext("given-names", "").strip()
            if surname and given:
                authors.append(f"{given} {surname}")
    return authors

def _parse_date(meta: ET.Element) -> date | None:
    candidates = [
        meta.find(".//date[@date-type='accepted']"),
        meta.find(".//date[@date-type='received']"),
        meta.find(".//pub-date[@pub-type='epub']"),
        meta.find(".//pub-date[@date-type='pub']"),
        meta.find(".//pub-date[@pub-type='ppub']")
    ]

    for el in candidates:
        if el is None:
            continue
        year = el.findtext("year")
        month = el.findtext("month")
        day = el.findtext("day")

        if not year:
            continue

        try:
            return date(
                int(year),
                int(month) if month else 1,
                int(day) if day else 1,
            )
        except ValueError:
            logger.warning(f"Bad date: {year}-{month}-{day}")

    return None

def _parse_abstract(meta: ET.Element) -> Section | None:
    abstract_el = meta.find("abstract")
    if abstract_el is None:
        logger.warning("No abstract element found")
        return
    
    sec, _, _ = _parse_section(sec=abstract_el)
    return sec

## body ##

def _parse_body(body: ET.Element | None) -> tuple[list[Section], list[Media], list[InlineTable]]:

    sections: list[Section] = []
    media: list[Media] = []
    tables: list[InlineTable] = []
    for sec in body.findall("./sec"):
        s, m, t = _parse_section(sec)
        if s: sections.append(s)
        if m: media.extend(m)
        if t: tables.extend(t)

    return sections, media, tables

## media and tables ##

def _parse_fig(fig: ET.Element) -> Media:
    fig_id = fig.get("id", "")

    label = _element_text(fig.find("label"))
    caption = _element_text(fig.find("caption"))

    graphic = fig.find("graphic")
    filename = ""

    if graphic is not None:
        href = graphic.get(_XLINK, "")
        filename = Path(href).name if href else ""

    return Media(
        id=fig_id,
        label=label,
        caption=caption,
        filename=filename,
    )

def _parse_table_wrap(table: ET.Element):
    table_id = table.get("id", "")
    label = _element_text(table.find("label"))
    caption = _element_text(table.find("caption"))

    media = []
    tables = []

    table_el = table.find("table")

    if table_el is not None:
        rows = _extract_table_rows(table_el)

        tables.append(
            InlineTable(
                headers=[],
                rows=rows,
                metadata={
                    "row_count": len(rows),
                    "column_count": max((len(r) for r in rows), default=0),
                },
            )
        )

    # optional embedded graphics inside table-wrap
    for g in table.findall("graphic"):
        href = g.get(_XLINK, "")
        media.append(
            Media(
                id=table_id,
                label=label,
                caption=caption,
                filename=Path(href).name if href else "",
            )
        )

    return media, tables

def _extract_table_rows(table: ET.Element) -> list[list[str]]:
    rows = []

    for tr in table.findall(".//tr"):
        row = [
            _element_text(cell).strip()
            for cell in tr
            if cell.tag in ("th", "td")
        ]
        if row:
            rows.append(row)

    return rows

## references ##

def _parse_references(back: ET.Element | None) -> list[Reference]:
    if back is None:
        return []

    out: list[Reference] = []

    for ref in back.findall(".//ref"):
        try:
            out.append(_parse_reference(ref))
        except ValueError as e:
            logger.warning(f"Bad reference: {e}")

    return out

def _parse_reference(ref: ET.Element) -> Reference | None:
    ref_id = ref.get("id", "")

    citation = ref.find(".//element-citation")
    if citation is None: citation = ref.find(".//mixed-citation")
    if citation is None:
        logger.warning(f"Skipping ref, could not find citation")
        return

    title_el = citation.find("article-title")
    if title_el is None:
        title_el = citation.find("chapter-title")
    title = _element_text(title_el)

    authors = []
    for name in citation.findall(".//string-name") + citation.findall(".//name"):
        surname = name.findtext("surname", "").strip()
        given = name.findtext("given-names", "").strip()
        if surname or given:
            authors.append(f"{given} {surname}".strip())

    year = citation.findtext("year", "").strip()
    year_val = int(year) if year.isdigit() else None

    doi = citation.findtext("pub-id[@pub-id-type='doi']")
    pmcid = citation.findtext("pub-id[@pub-id-type='pmcid']")
    return Reference(
        id=ref_id,
        title=title,
        authors=authors,
        pub_type=citation.get("publication-type", ""),
        source=_element_text(citation.find(".//source")),
        year=year_val,
        volume=citation.findtext("volume", "").strip(),
        page_start=citation.findtext("fpage", "").strip(),
        page_end=citation.findtext("lpage", "").strip(),
        doi=(doi.strip() if doi else ""),
        pmcid=(pmcid.strip() if pmcid else ""),
    )

## helpers ##

def _parse_section(sec: ET.Element) -> tuple[Section, list[Media], list[InlineTable]]:
    if sec.get("sec-type") == "supplementary-material":
        return None, [], []

    title_el = sec.find("title")
    header = _element_text(title_el)
    section_id = hash(header)

    content: list[Paragraph | Section | List] = []
    media: list[Media] = []
    tables: list[InlineTable] = []
    for child in sec:
        tag = child.tag
        if tag in _TITLE_TAGS:
            continue
        elif tag == "sec":
            sub_sec, m, t = _parse_section(child)
            if sub_sec: content.append(sub_sec)
            media.extend(m)
            tables.extend(t)
        elif tag == "p":
            sentences = _extract_sentences(child)
            if sentences:
                p = Paragraph(
                    id="p_" + hashlib.md5("".join(s.text for s in sentences).encode()).hexdigest()[:8],
                    sentences=sentences
                )
                content.append(p)
        elif tag == "fig":
            m = _parse_fig(child)
            if m: media.append(m)
        elif tag == "table-wrap":
            m, t = _parse_table_wrap(child)
            if m: media.extend(m)
            if t: tables.extend(t)
        elif tag == "list":
            list = _parse_list(child)
            if list: content.append(list)

    if not content and not media and not tables:
        return None, [], []

    return Section(
        id=section_id,
        header=header,
        content=content,
    ), media, tables

def _parse_list(list: ET.Element) -> List:
    bullets = []
    for list_item in list:
        sentences = _extract_sentences(p=list_item)
        if sentences: bullets.append(sentences)
    return List(items=bullets)

_SENTINEL = "\x00{}\x01{}\x00"
_SENTINEL_RE = re.compile(r"\x00([^\x00\x01]+)\x01([^\x00\x01]+)\x00")

def _expand_rid_range(start_rid: str, end_rid: str) -> list[str]:
    prefix  = re.match(r"[^\d]+", start_rid).group()
    start_n = int(re.search(r"\d+", start_rid).group())
    end_n   = int(re.search(r"\d+", end_rid).group())
    return [f"{prefix}{i}" for i in range(start_n, end_n + 1)]

def _extract_sentences(p: ET.Element) -> list[Sentence]:

    # collect all text inside p, placing a sentinel for every xref
    parts = []
    def collect(e: ET.Element) -> None:
        if e is None:
            return
        if e.text: parts.append(e.text)
        children = list(e)
        i = 0
        while i < len(children):
            child = children[i]
            if child.tag == "xref" and (child.get("ref-type", "") in {"bibr", "fig", "table"}):
                if child.text: parts.append(child.text)
                ref_type = child.get("ref-type", "")
                rid = child.get("rid", "")
                
                # check if range citation
                next_child = children[i + 1] if i + 1 < len(children) else None
                if ((child.tail or "").strip() in _RANGE_DASHES and
                    next_child is not None and
                    next_child.tag == "xref" and
                    next_child.get("ref-type") == "bibr"):
                    
                    parts.append(child.tail.strip())
                    parts.append(next_child.text)
                    
                    next_rid = next_child.get("rid", "")
                    for rid in _expand_rid_range(rid, next_rid):
                        parts.append(_SENTINEL.format(ref_type, rid))
                    
                    if next_child.tail: parts.append(next_child.tail)
                    i += 2 # skip next_child
                else:
                    # not a range citation
                    parts.append(_SENTINEL.format(ref_type, rid))
                    if child.tail: parts.append(child.tail)
                    i += 1
            else:
                collect(child)
                if child.tail: parts.append(child.tail)
                i += 1
    collect(p)
    flat_text = re.sub(r"\s+", " ", "".join(parts)).strip()

    # split flat_text into sentences
    out = []
    sentences = _tokenizer.tokenize(flat_text)
    for sentence in sentences:
        # get InlineRefs
        refs=[]
        for m in _SENTINEL_RE.finditer(sentence):
            ref_type = m.group(1)
            rid = m.group(2)
            refs.append(InlineRef(type=ref_type, target=rid))
        
        # remove sentinels from sentence
        clean_text = _SENTINEL_RE.sub("", sentence)
        if not clean_text: continue

        out.append(Sentence(id=hash(clean_text), text=clean_text, refs=refs))

    return out

def _element_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"\s+", " ", "".join(parts)).strip()

if __name__ == "__main__":
    import json
    from dataclasses import asdict

    test_file = "/Users/karenxu/Documents/Code/USRA/datasets/papers/1cc57bae-7c43-1014-843f-f407711b0a12/paper/720970.xml"
    paper = parse_file(test_file)

    with open("test.json", "w") as f:
        json.dump(asdict(paper), f, ensure_ascii=False)

    full_text = paper.to_markdown()
    with open("test.md", "w") as f:
        f.write(full_text)