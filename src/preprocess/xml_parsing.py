from __future__ import annotations

import re
import logging
from pathlib import Path
from datetime import date
import xml.etree.ElementTree as ET
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters
from utils.schemas import (
    ContentItem,
    TextSpan,
    TableSpan,
    MediaSpan,
    ListSpan,
    Paper,
    Section,
    Reference,
    PaperBack,
    PaperFront
)

punkt_params = PunktParameters()
punkt_params.abbrev_types = {"fig", "figs", "et al", "e.g", "i.e", "vs", "c.a", "approx", "dept"}
tokenizer = PunktSentenceTokenizer(punkt_params)

logger = logging.getLogger(__name__)

class MissingContentError(ValueError):
    '''raise when expected xml elements are missing'''

_XLINK = "{http://www.w3.org/1999/xlink}href"
_MEDIA_TAGS = {"graphic", "media", "inline-graphic", "fig"}
_TITLE_TAGS = {"title", "label"}

## main ##

def parse_file(path: Path) -> Paper:
    logger.info("Parsing {path}...")
    tree = ET.parse(path)
    root = tree.getroot()
    article = root.find("article")
    return _parse_article(article) if article is not None else _parse_article(root)

def parse_string(xml_text: str) -> Paper:
    root = ET.fromstring(xml_text)
    article = root.find("article")
    return _parse_article(article) if article is not None else _parse_article(root)

## main helper ##

def _parse_article(article: ET.Element) -> Paper:
    front_el = article.find("front")
    body_el = article.find("body")
    back_el = article.find("back")

    front = _parse_front(front_el)
    back = _parse_back(back_el)
    body = _parse_body(body_el)

    return Paper(front=front, back=back, body=body)

## parse front matter ##

def _parse_front(front: ET.Element | None) -> PaperFront:
    if front is None: raise MissingContentError("No front tag found")

    meta = front.find("article-meta")
    if meta is None: raise MissingContentError("No meta tag found")
    
    title = _find_title(meta)
    doi = _find_pub_id(meta, "doi")
    authors = _parse_authors(meta)
    keywords = _parse_keywords(meta)
    categories = _parse_categories(meta)
    pub_date = _parse_date(meta)
    abstract = _parse_abstract(meta)

    return PaperFront(
        title=title,
        doi=doi,
        abstract=abstract,
        keywords=keywords,
        authors=authors,
        date=pub_date,
        categories=categories,
    )

def _find_title(meta: ET.Element) -> str:
    el = meta.find(".//article-title")
    if el is not None: return _element_text(el)
    logger.warning("No article-title found")
    return ""

def _find_pub_id(meta: ET.Element, id_type: str) -> str:
    for el in meta.findall("article-id"):
        if el.get("pub-id-type") == id_type:
            return el.text.strip()
    logger.warning("No pub-id found")
    return ""

def _parse_authors(meta: ET.Element) -> list[str]:
    authors = []
    for contrib in meta.findall(".//contrib[@contrib-type='author']"):
        name_el = contrib.find("name")
        if name_el is not None:
            surname = name_el.findtext("surname", "").strip()
            given = name_el.findtext("given-names", "").strip()
            if surname and given:
                authors.append(f"{surname}, {given}".strip())
            else:
                logger.warning("Could not find surname and given name for author.")
        else:
            logger.warning("Could not find name tag in contrib")
    
    return authors

def _parse_keywords(meta: ET.Element) -> list[str]:
    keywords = [
        kwd.text.strip()
        for kwd in (meta.findall(".//kwd"))
        if kwd.text
    ]

    if not keywords:
        logger.warning("No keywords found")

    return keywords

def _parse_categories(meta: ET.Element) -> list[str]:
    categories= [
        subj.text.strip()
        for subj in (meta.findall(".//subject"))
        if subj.text
    ]

    if not categories:
        logger.warning("No categories found")
    
    return categories

def _parse_date(meta: ET.Element) -> date | None:
    '''
    gets epub date, otherwise gets date received
    '''
    epub_el = meta.find(".//pub-date[@pub-type='epub']")
    date_el = meta.find(".//date[@date-type='received']")

    for el in [epub_el, date_el]:
        if el is None: continue
        year = el.findtext("year")
        month = el.findtext("month", "1")
        day = el.findtext("day", "1")
        if year:
            try:
                return date(int(year), int(month), int(day))
            except ValueError:
                logger.warning(f"Could not form date from {year}, {month}, {day}")
        else:
            logger.warning("No year found")

    logger.warning("No epub or received date found")
    return None

def _parse_abstract(meta: ET.Element) -> Section:
    abstract_el = meta.find("abstract")
    if abstract_el is None:
        raise MissingContentError(f"No abstract tag found")
    return _parse_section(abstract_el)

## body ##

def _parse_body(body: ET.Element | None) -> list[Section]:
    if body is None:
        raise MissingContentError("No body tag found")
    all_secs: list[Section] = []
    for sec in body.findall("sec"):
        sec: Section = _parse_section(sec)
        if sec: all_secs.append(sec)

    return all_secs

def _parse_section(sec: ET.Element) -> Section | None:
    if sec.get("sec-type", "") == "supplementary-material": return None

    # get section title
    title_el = sec.find("title")
    if title_el is None: logger.warning("No title tag found in sec")
    section_title = _element_text(title_el)

    # process section children
    sec_items = []
    for child in sec:
        tag = child.tag
        if tag in _TITLE_TAGS: continue
        if tag == "sec":
            item: Section = _parse_section(child)
            if item: sec_items.append(item)
        elif tag == "p":
            items: list[ContentItem] = _parse_paragraph_element(child)
            if items: sec_items.extend(items)
        elif tag in _MEDIA_TAGS:
            item: MediaSpan = _parse_fig(child)
            if item: sec_items.append(item)
        elif tag == "table-wrap":
            items: list[TableSpan | MediaSpan] = _parse_table_wrap(child)
            if items: sec_items.extend(items)
        elif tag == "list":
            item: ListSpan = _parse_list(child)
            if item: sec_items.append(item)
    
    return Section(header=section_title, content=sec_items)

## paragraph / in-line content ##

def _parse_paragraph_element(el: ET.Element) -> list[TextSpan]:

    # collect text and xrefs
    texts: list[str] = []
    xrefs: dict[str, str] = {} # {rid: ref_type}
    def collect(node: ET.Element) -> None:
        # collect text before first child tag
        if node.text: texts.append(node.text.strip())
        for child in node:
            if child.tag == "xref" and child.get("ref-type") in {"bibr", "fig", "table"}:
                # append xref inner text to texts
                if child.text: texts.append(child.text.strip())

                rid = child.get("rid", "")
                if not rid: logger.warning("No rid found")
                ref_type = child.get("ref-type")

                # store temp marker
                texts.append(f"###{rid}###")

                # store ref info
                xrefs[rid] = ref_type
                
                if child.tail: texts.append(child.tail.strip())
            elif child.tag != "sup":
                collect(child)
                if child.tail: texts.append(child.tail.strip())
            else:
                if child.tail: texts.append(child.tail.strip())
    collect(el)

    # convert texts to flat text with xref markers
    normalised_texts = [ " ".join(text.split()) for text in texts]
    flat_text = " ".join(normalised_texts)

    # split full text on sentences
    sentences = tokenizer.tokenize(flat_text)

    # for each sentence, create TextSpan
    text_spans: list[TextSpan] = []
    for s in sentences:
        matches = re.findall(r"###([^#]+)###", s)
        # remove temp markers from text
        stripped_s = " ".join(re.sub(r"###[^#]+###", "", s).split())
        table_ids = [rid for rid in matches if xrefs[rid] == "table"]
        fig_ids   = [rid for rid in matches if xrefs[rid] == "fig"]
        ref_ids   = [rid for rid in matches if xrefs[rid] == "bibr"]

        text_spans.append(TextSpan(text=stripped_s, table_ids=table_ids, fig_ids=fig_ids, ref_ids=ref_ids))
        
    return text_spans

def _parse_fig(fig: ET.Element) -> MediaSpan:
    fig_id = fig.get("id", "")

    caption = fig.find("caption")
    caption_text = _element_text(caption).strip()

    label = fig.find("label")
    label_text = _element_text(label).strip()

    graphic_el = fig.find(".//graphic")
    if graphic_el is not None:
        href = graphic_el.get(_XLINK, "")
        if not href:
            raise MissingContentError(f"No xlink:href on graphic in {fig.tag}")
        img_name = Path(href).name
        return MediaSpan(media_id=fig_id, label=label_text, caption=caption_text, name=img_name)

    logger.warning("No graphic tag found in fig")
    return MediaSpan(media_id=fig_id, label=label_text, caption=caption_text, name=None)

def _parse_table_wrap(wrap: ET.Element) -> list[MediaSpan | TableSpan]:
    table_id = wrap.get("id", "")

    caption = wrap.find("caption")
    caption_text = _element_text(caption).strip()

    label = wrap.find("label")
    label_text = _element_text(label).strip()

    # check if graphic
    media_spans = []
    graphics = wrap.findall("graphic")
    for graphic in graphics:
        href = graphic.get(_XLINK, "")
        if not href: raise MissingContentError(f"No xlink:href on graphic")
        img_name = Path(href).name
        media_spans.append(MediaSpan(media_id=table_id, label=label_text, caption=caption_text, name=img_name))
    if media_spans: return media_spans

    # check if xml table
    table = wrap.find("table")
    if table is not None:
        md = _table_to_markdown(table)
        if md:
            return [TableSpan(markdown=md, label=label_text, caption=caption_text, table_id=table_id)]

    logger.warning("No graphic or table found in table-wrap")
    return [TableSpan(caption=caption_text, label=label_text, table_id=table_id, markdown=None)]

def _table_to_markdown(table: ET.Element) -> str:
    rows: list[list[str]] = []

    for tr in table.findall(".//tr"):
        cells = []
        for cell in tr:
            if cell.tag in ("th", "td"):
                cells.append(_element_text(cell).replace("\n", " ").strip())
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    col_widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(width)]

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[j]) for j, c in enumerate(cells)) + " |"

    lines = [fmt_row(rows[0])]
    lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)

def _parse_list(list_el: ET.Element) -> ListSpan:
    items: list[ContentItem]= []
    for item in list_el.findall("list-item"):
        for p in item.findall("p"):
            sub_items = _parse_paragraph_element(p)
            items.extend(sub_items)
    return ListSpan(list_items=items)

## back matter ##

def _parse_back(back: ET.Element | None) -> PaperBack:
    if back is None:
        raise MissingContentError("No back tag found")

    references = _parse_references(back)

    return PaperBack(references=references)

def _parse_references(back: ET.Element) -> list[Reference]:
    refs = []
    for ref in back.findall(".//ref"):
        refs.append(_parse_reference(ref))
    return refs

def _parse_reference(ref: ET.Element) -> Reference:
    ref_id = ref.get("id", "")
    citation = ref.find("element-citation")
    if citation is None:
        citation = ref.find("mixed-citation")

    if citation is None:
        return Reference(ref_id=ref_id, raw_text=_element_text(ref))

    pub_type = citation.get("publication-type", "")
    doi = ""
    pmid = ""
    pmcid = ""
    for pub_id in citation.findall("pub-id"):
        id_type = pub_id.get("pub-id-type", "")
        val = pub_id.text or ""
        if id_type == "doi":
            doi = val.strip()
        elif id_type == "pmid":
            pmid = val.strip()
        elif id_type == "pmcid":
            pmcid = val.strip()

    title_el = citation.find("article-title")
    if title_el is None:
        title_el = citation.find("chapter-title")
    title = _element_text(title_el) if title_el is not None else ""

    authors = []
    search_group = citation.findall(".//string-name") or citation.findall(".//name")
    for name in search_group:
        surname = name.findtext("surname", "")
        given = name.findtext("given-names", "")
        if surname and given:
            authors.append(f"{surname}, {given}".strip())
        else:
            logger.warning("Could not find surname and given name for author.")

    return Reference(
        ref_id=ref_id,
        authors=authors,
        title=title,
        pub_type=pub_type,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
    )

## helpers ##

def _element_text(el: ET.Element | None) -> str:
    '''
    recursively collects all text in an element ignoring tags
    '''
    if el is None:
        return ""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(" ".join(parts).split())