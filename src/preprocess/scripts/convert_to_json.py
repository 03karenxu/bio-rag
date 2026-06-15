from preprocess.xml_parsers import elsevier
from preprocess.xml_parsers import jats
from preprocess.xml_parsers.schema import *
from utils.logging import init_logging
from config import DATASET_DIR
import argparse
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    init_logging("convert_to_json.log")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--in-dir", default="lit_reviews_test", help="Directory to read from")
    arg_parser.add_argument("--out-dir", default="lit_reviews_json", help="Directory to save json files")
    args = arg_parser.parse_args()

    in_dir = DATASET_DIR / args.in_dir
    if not in_dir.exists():
        raise FileNotFoundError(f"{in_dir} not found")

    out_dir = DATASET_DIR / args.out_dir
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    
    xml_files = list(in_dir.rglob("*.xml"))
    failed = 0
    for xml_file in tqdm(xml_files):
        try:
            parsed = elsevier.parse_file(xml_file) if xml_file.name.startswith("10") else jats.parse_file(xml_file)
        except Exception as e:
            logger.warning(f"Failed to parse {xml_file}: {e}")
            failed += 1
            continue
        out_path = out_dir / xml_file.parent.parent.name / xml_file.parent.name / xml_file.with_suffix(".json").name
        out_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            with open(out_path, "w") as f:
                json.dump(parsed.model_dump(), f)
                logger.info(f"Created file {out_path}")
        except Exception as e:
            logger.warning(f"Failed to save {out_path}: {e}")
            failed += 1
    
    logger.info(f"Done! Failed to convert {failed} files")

            
            
            
