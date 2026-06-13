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
    el = root.find(".//ja:article", namespaces=_NAMESPACES)
    return _parse_article(el)

def parse_string(xml_text: str) -> Paper:
    root = ET.fromstring(xml_text)
    el = root.find(".//ja:article", namespaces=_NAMESPACES)
    return _parse_article(el)

# ------------------------------------------------------------------------------

## private globals ##

_TITLE_TAGS = {"label", "section-title"}

_punkt_params = PunktParameters()
_punkt_params.abbrev_types = {
    "fig", "figs", "et al", "e.g", "i.e", "vs", "c.a", "approx", "dept", "e", "no"
}
_tokenizer = PunktSentenceTokenizer(_punkt_params)

_NAMESPACES = {
    # Elsevier delivery / wrapper layer
    "xocs": "http://www.elsevier.com/xml/xocs/dtd",
    "xoe": "http://www.elsevier.com/xml/xoe/dtd",

    # Core Elsevier article structure (most important)
    "ce": "http://www.elsevier.com/xml/common/dtd",
    "sb": "http://www.elsevier.com/xml/common/struct-bib/dtd",

    # Linking (figures, graphics, URLs)
    "xlink": "http://www.w3.org/1999/xlink",

    # Metadata / publishing standards
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",

    # Article content layer (JA / full-text article)
    "ja": "http://www.elsevier.com/xml/ja/dtd",
}

_SENTINEL = "\x00{}\x01{}\x00"
_SENTINEL_RE = re.compile(r"\x00([^\x00\x01]+)\x01([^\x00\x01]+)\x00")

## top-level ##

def _parse_article(article: ET.Element) -> Paper:
    head_el = article.find(".//ja:head", namespaces=_NAMESPACES)
    if head_el is None: raise MissingContentError(element="head", e=article)
    front = _parse_front(head_el)

    body = article.find(".//ja:body", namespaces=_NAMESPACES)
    if body is None: raise MissingContentError(element="body", e=article)
    sections, media, tables = _parse_body(body)

    tail = article.find(".//ja:tail", namespaces=_NAMESPACES)
    references = _parse_references(tail)

    return Paper(
        front=front,
        body=sections,
        media=media,
        inline_tables=tables,
        references=references,
    )

## front ##

def _parse_front(head: ET.Element) -> Front:

    # get article title
    title_el    = head.find(".//ce:title", namespaces=_NAMESPACES)
    title       = _element_text(title_el) if title_el is not None else ""

    authors     = _parse_authors(head)
    pub_date    = _parse_date(head)
    abstract    = _parse_abstract(head)
    return Front(
        title=title,
        authors=authors,
        hash=hashlib.md5(title.encode()).hexdigest().upper(),
        publication_date=pub_date.isoformat() if pub_date else "",
        abstract=abstract,
    )

def _parse_authors(head: ET.Element) -> list[str]:
    authors = []
    author_group_el = head.find(".//ce:author-group", namespaces=_NAMESPACES)
    for author_el in author_group_el:
        surname = author_el.findtext(".//ce:surname", "", namespaces=_NAMESPACES).strip()
        given   = author_el.findtext(".//ce:given-name", "", namespaces=_NAMESPACES).strip()
        if surname and given:
            authors.append(f"{given} {surname}")
    return authors

def _parse_date(head: ET.Element) -> date | None:
    date_acc_el = head.find(".//ce:date-accepted", namespaces=_NAMESPACES)
    if date_acc_el is None:
        logger.warning("No date accepted element found")
        return
    
    year    = date_acc_el.get("year", "").strip()
    month   = date_acc_el.get("month", "").strip()
    day     = date_acc_el.get("day", "").strip()

    try:
        return date(
            int(year),
            int(month),
            int(day),
        )
    except ValueError:
        logger.warning(f"Bad date: {year}-{month}-{day}")
        return

def _parse_abstract(head: ET.Element) -> Section | None:
    abstract_el = head.find(".//ce:abstract", namespaces=_NAMESPACES)
    if abstract_el is None:
        logger.warning("Could not find abstract")
        return

    sections: list[Section] = []
    for sec in abstract_el.findall(".//ce:abstract-sec", namespaces=_NAMESPACES):
        section_title = sec.findtext("ce:section-title", default="", namespaces=_NAMESPACES)
        paragraphs: list[Paragraph] = []
        for para in sec.findall(".//ce:simple-para", namespaces=_NAMESPACES):
            sentences = _extract_sentences(para)
            if not sentences: continue
            para_id = para.get("id") or _stable_id("".join(s.text for s in sentences))
            paragraphs.append(
                Paragraph(
                    id=para_id,
                    sentences=sentences,
                )
            )
        if paragraphs:
            sections.append(
                Section(
                    id=_stable_id(section_title + sec.get("id", "")),
                    header=section_title,
                    content=paragraphs,
                )
            )

    # fallback: flat abstract
    if not sections:
        paragraphs = []
        for para in abstract_el.findall(".//ce:simple-para", namespaces=_NAMESPACES):
            sentences = _extract_sentences(para)
            if sentences:
                paragraphs.append(
                    Paragraph(
                        id=para.get("id") or _stable_id("".join(s.text for s in sentences)),
                        sentences=sentences,
                    )
                )
        return Section(
            id=_stable_id("abstract"),
            header="Abstract",
            content=paragraphs,
        )

    if len(sections) == 1:
        s = sections[0]
        s.header = "Abstract"
        return s
    
    return Section(
        id=_stable_id("abstract"),
        header="Abstract",
        content=sections,
    )

## body ##

def _parse_body(body: ET.Element | None) -> tuple[list[Section], list[Media], list[InlineTable]]:
    sections: list[Section] = []
    media: list[Media] = []
    tables: list[InlineTable] = []
    for sec in body.findall(".//ce:section", namespaces=_NAMESPACES):
        s = _parse_section(sec)
        if s: sections.append(s)

    return sections, media, tables

## media and tables ##

def _parse_fig(fig: ET.Element) -> Media:
    ...

def _parse_table_wrap(table: ET.Element):
    ...

def _extract_table_rows(table: ET.Element) -> list[list[str]]:
    ...

## references ##

def _parse_references(tail: ET.Element | None) -> list[Reference]:
    if tail is None:
        logger.warning("No tail element found")
        return []

    out: list[Reference] = []
    for ref in tail.findall(".//ce:bib-reference", namespaces=_NAMESPACES):
        try:
            out.append(_parse_reference(ref))
        except ValueError as e:
            logger.warning(f"Bad reference: {e}")

    return out

def _parse_reference(ref: ET.Element) -> Reference:
    ref_id = ref.get("id", "")

    citation = ref.find(".//sb:reference", namespaces=_NAMESPACES)
    if citation is None:
        logger.warning(f"Skipping ref, could not find citation")
        return
    
    title_el = citation.find(".//sb:maintitle", namespaces=_NAMESPACES)
    title = _element_text(title_el)

    authors = []
    for author in citation.findall(".//sb:author", namespaces=_NAMESPACES):
        surname = author.findtext(".//ce:surname", "", namespaces=_NAMESPACES).strip()
        given = author.findtext(".//ce:given-name", "", namespaces=_NAMESPACES).strip()
        if surname and given:
            authors.append(f"{given} {surname}".strip())

    year = citation.findtext(".//sb:date", "", namespaces=_NAMESPACES).strip()
    year_val = int(year) if year.isdigit() else None

    return Reference(
        id=ref_id,
        title=title,
        authors=authors,
        pub_type=citation.get("publication-type", ""),
        source=_element_text(citation.find("./sb:host/sb:issue/sb:series/sb:title", namespaces=_NAMESPACES)),
        year=year_val,
        volume=citation.findtext(".//sb:volume-nr", "", namespaces=_NAMESPACES).strip(),
        page_start=citation.findtext(".//sb:first-page", "", namespaces=_NAMESPACES).strip(),
        page_end=citation.findtext(".//sb:last-page", "", namespaces=_NAMESPACES).strip(),
        doi="",
        pmcid=""
    )

## helpers ##

def _parse_section(sec: ET.Element) -> Section:
    title_el = sec.find("ce:section-title", namespaces=_NAMESPACES)
    header = _element_text(title_el)
    section_id = sec.get("id", _stable_id(header))

    content: list[Paragraph | Section | List] = []
    for child in sec:
        tag = _local(child.tag).strip()
        if tag in _TITLE_TAGS:
            continue
        elif tag == "section":
            sub_sec = _parse_section(child)
            if sub_sec: content.append(sub_sec)
        elif tag == "para":
            sentences = _extract_sentences(child)
            id = child.get("id", "p_" + hashlib.md5("".join(s.text for s in sentences).encode()).hexdigest()[:8])
            if sentences:
                p = Paragraph(
                    id=id,
                    sentences=sentences
                )
                content.append(p)

    if not content: return None

    return Section(
        id=section_id,
        header=header,
        content=content,
    )

def _get_reftype(rid: str) -> str | None:
    if rid.startswith("tbl"):
        return "table"
    elif rid.startswith("bib"):
        return "bibr"
    elif rid.startswith("fig"):
        return "fig"
    return

def _extract_sentences(p: ET.Element) -> list[Sentence]:
    # collect all text inside p, placing a sentinel for every xref
    parts = []
    def collect(e: ET.Element) -> None:
        if e is None:
            return
        if e.text: parts.append(e.text)
        for child in e:
            tag = _local(child.tag).strip()
            if tag == "cross-ref":
                # single cross ref
                parts.append(_element_text(child))
                rid = child.get("refid", "")
                ref_type = _get_reftype(rid)
                if not ref_type:
                    if child.tail: parts.append(child.tail)
                    continue

                parts.append(_SENTINEL.format(ref_type, rid))
                if ref_type == "bibr": parts.append(". ")
                if child.tail: parts.append(child.tail)
            elif tag == "cross-refs":
                # multiple refs
                parts.append(_element_text(child))
                rids = child.get("refid").split()
                for rid in rids:
                    ref_type = _get_reftype(rid)
                    if not ref_type:    
                        if child.tail: parts.append(child.tail)
                        continue
                    parts.append(_SENTINEL.format(ref_type, rid))
                if ref_type == "bibr": parts.append(". ")
                if child.tail: parts.append(child.tail)
            else:
                collect(child)
                if child.tail: parts.append(child.tail)

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

def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag

def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

if __name__ == "__main__":
    import json
    from dataclasses import asdict

    test_file = "/Users/karenxu/Documents/Code/USRA/datasets/10.1016_j.archoralbio.2010.06.016.xml"
    paper = parse_file(test_file)

    with open("test.json", "w") as f:
        json.dump(asdict(paper), f, ensure_ascii=False)

    full_text = paper.to_markdown()
    with open("test.md", "w") as f:
        f.write(full_text)