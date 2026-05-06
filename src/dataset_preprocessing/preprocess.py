from __future__ import annotations
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from pydantic import BaseModel

from paths import DATASET_DIR

logging.getLogger(__name__)

MIN_PARAGRAPH_SIZE = 20 # min character length
TEST_XML = "/Users/karenxu/Documents/Code/USRA/dataset/papers/9f41b294-6e40-1014-b16d-ff0e7a239a97/721408.xml"

class Section(BaseModel):
    title: str
    children: list[str] | list[Section]

class Paper(BaseModel):
    title: str
    doi: str
    abstract: str
    keywords: list[str]
    authors: list[str]
    date: str
    categories: list[str]
    sections: list[Section]


def get_all_text(element, path: str) -> str:
    '''
    gets all the text inside an element, including the text inside child elements
    '''
    e = element.find(path)
    if e is None:
        return ""
    return "".join(e.itertext()).strip()


def parse_paper(xml_path: Path) -> Paper:
    '''
    unpacks relevant information from a paper's xml file into a Paper model
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    def get_date(date_type: str) -> str:
        date = root.find(f".//history/date[@date-type='{date_type}']")
        if date is None:
            return ""
        year = date.findtext("year", default="")
        month = date.findtext("month", default="").zfill(2)
        day = date.findtext("day", default="").zfill(2)
        return f"{year}-{month}-{day}"

    def get_abstract() -> str:
        abstract = root.find(".//abstract")
        if abstract is None:
            return ""
        parts = []
        for child in abstract:
            if child.tag != "title":
                parts.extend(child.itertext())
        return "".join(parts).strip()


    def get_sections() -> list[Section]:
        def parse_section(sec) -> Section:
            title = get_all_text(sec, "title")

            # handle subsections
            child_secs = sec.findall("sec")
            if child_secs:
                return Section(
                    title=title,
                    children=[parse_section(s) for s in child_secs]
                )
            
            paragraphs = []
            for child in sec:
                if child.tag not in ("sec", "title"):
                    text = "".join(child.itertext()).strip()
                    # filter out paragraphs with less than threshold
                    if text and len(text) > MIN_PARAGRAPH_SIZE:
                        paragraphs.append(text)
            return Section(title=title, children=paragraphs)
    
        return [
            s for sec in root.findall(".//body/sec")
            if (s := parse_section(sec)).children
        ]
    
    return Paper(
        title=get_all_text(root, ".//article-title"),
        doi=get_all_text(root, ".//article-id[@pub-id-type='doi']"),
        abstract=get_abstract(),
        keywords=["".join(kwd.itertext()).strip() for kwd in root.findall(".//kwd") if "".join(kwd.itertext()).strip()],
        authors=[
            f"{get_all_text(c, 'name/surname')}, {get_all_text(c, 'name/given-names')}".strip(", ")
            for c in root.findall(".//contrib[@contrib-type='author']")
        ],
        date=get_date("accepted") or get_date("received"),
        categories=[s.text for s in root.findall(".//subj-group/subject") if s.text],
        sections=get_sections()
    )

if __name__ == "__main__":
    paper = parse_paper(Path(TEST_XML))

    with open("xml.json", "w") as f:
        f.write(paper.model_dump_json())