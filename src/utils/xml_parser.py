from __future__ import annotations
 
import re
import tiktoken
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET
from utils.models import Paper, Chunk
from utils.image_handling import estimate_image_tokens, get_image_paths
from config import MIN_CHUNK_TOKENS, COHERE_TRANSFORMABLE_FORMATS, COHERE_COMPATIBLE_FORMATS

_MEDIA_L = "[[["
_MEDIA_R = "]]]"
MEDIA_MARKER = re.compile(re.escape(_MEDIA_L) + r'(.+?)' + re.escape(_MEDIA_R))

_XLINK = "{http://www.w3.org/1999/xlink}href"
_MEDIA_TAGS = {"graphic", "media", "inline-graphic"}
_TITLE_TAGS = {"title", "label"}

class PaperParser:
    '''
    parses a preprint xml file from biorxiv into a Paper model
    '''

    def __init__(self, xml: Path, token_enc_type: str = "cl100k_base"):
        self.root = ET.parse(xml).getroot()
        self.paper_dir = xml.parent
        self.TT = tiktoken.get_encoding(token_enc_type)

    def parse_paper(self) -> Paper:
        '''
        unpacks relevant information from a paper's XML file into a Paper model.
        accepts either an XML string or a path to an XML file.
        '''
        
        return Paper(
            title=self.get_title(),
            doi=self.get_doi(),
            abstract=self.get_abstract(),
            keywords=self.get_keywords(),
            authors=self.get_authors(),
            date=self.get_date("accepted") or self.get_date("received"),
            categories=self.get_categories(),
            body=self.get_body(),
        )

    def get_title(self) -> str:
        e = self.root.find("front/article-meta/title-group/article-title") or self.root.find(".//article-title")
        return self._get_all_text_with_media(e)
    
    def get_doi(self) -> str:
        e = (
            self.root.find("./front/article-meta/article-id[@pub-id-type='doi']") or
            self.root.find(".//article-id[@pub-id-type='doi']")
        )
        return self._get_all_text_with_media(e)
    
    def get_abstract(self) -> list[Chunk]:
        abstract = self.root.find(".//abstract")
        return self._merge_small_chunks(self._process_section(abstract))

    def get_categories(self) -> list[str]:
        return [s.text for s in self.root.findall(".//subj-group/subject") if s.text]

    def get_keywords(self) -> list[str]:
        return [t for kwd in self.root.findall(".//kwd") if (t := self._get_all_text_with_media(kwd))]
    
    def get_authors(self) -> list[str]:
        return [
            f"{self._get_all_text_with_media(c.find('name/surname'))}, {self._get_all_text_with_media(c.find('name/given-names'))}".strip(", ")
            for c in self.root.findall(".//contrib[@contrib-type='author']")
        ]

    def get_date(self, date_type: str) -> date | None:
        node = self.root.find(f".//history/date[@date-type='{date_type}']")
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
    
    def get_body(self) -> list[Chunk]:
        body = self.root.find(".//body")
        return self._merge_small_chunks(self._process_section(body))

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

    def _process_section(self, sec: ET.Element, parent_title: str | None = None) -> list[Chunk]:
        title_elem = sec.find("title")
        title = " ".join(self._get_all_text_with_media(title_elem).split()).strip()
        full_title = f"{parent_title} > {title}" if parent_title else title

        chunks = []
        for child in sec:
            if child.tag == "sec":
                chunks.extend(self._process_section(child, full_title))
            elif child.tag not in _TITLE_TAGS:
                text = " ".join(self._get_all_text_with_media(child).split())
                if text:
                    full_text = f"Section: {full_title} Content: {text}"
                    n_tokens = len(self.TT.encode(full_text)) + sum(
                        estimate_image_tokens(path)
                        for m in MEDIA_MARKER.finditer(text)
                        for path in get_image_paths(self.paper_dir / m.group(1))
                    )
                    chunks.append(Chunk(n_tokens=n_tokens, section=full_title, text=text))
        return chunks

    def _get_all_text_with_media(self, e: ET.Element) -> str:
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
                parts.append(f"{_MEDIA_L}{Path(child.attrib[_XLINK]).name}{_MEDIA_R}")
            else:
                parts.append(self._get_all_text_with_media(child))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)


