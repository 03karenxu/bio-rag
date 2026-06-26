from pathlib import Path
from config import DATASET_DIR
from ingest.xml_parsers.jats import parse_file
import json
import requests

def pdf_to_jats(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        response = requests.post(
            "http://localhost:8070/api/processFulltextDocument",
            headers={"Accept": "application/vnd.jats+xml"},
            files={"file": (pdf_path.name, f, "application/pdf")},
        )
    response.raise_for_status()
    return response.text

def process_pdfs(dir: Path) -> None:
    if not Path(dir).exists():
        raise FileNotFoundError(f"{dir} not found")

    for paper_dir in Path(dir).iterdir():
        for pdf_path in paper_dir.rglob("*.pdf"):
            try:
                jats_xml = pdf_to_jats(pdf_path)
                xml_path = pdf_path.with_suffix(".xml")
                xml_path.write_text(jats_xml)
            except Exception as e:
                print(f"Failed {pdf_path}: {e}")

def process_jats(dir: Path, out_dir: Path) -> list[Path]:
    failed = []
    for paper_dir in Path(dir).iterdir():
        # parse main paper .xml
        main_paper = next((paper_dir/"paper").glob("*.xml"), None)
        if not main_paper: raise ValueError(f"No .xml file in {paper_dir/"paper"}")
        try:
            parsed = parse_file(main_paper)
            json_path = out_dir / main_paper.stem / "paper" / main_paper.with_suffix(".json").name
            json_path.parent.mkdir(exist_ok=True, parents=True)
            with open(json_path, "w") as f:
                json.dump(parsed, f)
        except Exception as e:
            failed.append(main_paper)
            print(f"Failed {main_paper}: {e}")

        # parse paper ref .xmls
        for file in (paper_dir / "refs").iterdir():
            if Path(file).suffix == ".xml":
                try:
                    parsed = parse_file(file)
                    json_path = out_dir / main_paper.stem / "refs" / file.with_suffix(".json").name
                    json_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(json_path, "w") as f:
                        json.dump(parsed, f)
                except Exception as e:
                    failed.append(file)
                    print(f"Failed {file}: {e}")

if __name__ == "__main__":
    # process_pdfs(DATASET_DIR / "lit_reviews")
    process_jats(DATASET_DIR / "lit_reviews", DATASET_DIR / "lit_reviews_json")
    
