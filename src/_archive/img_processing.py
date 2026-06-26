import math
import logging
from PIL import Image
from pathlib import Path
from config import COHERE_TRANSFORMABLE_FORMATS, COHERE_COMPATIBLE_FORMATS

logger = logging.getLogger(__name__)

def estimate_image_tokens(path: Path, detail: str = "high") -> int:
    '''
    estimates the number of tokens from an image path. assumes image is already
    resized to fit cohere max size
    '''
    if detail == "low":
        return 256
    with Image.open(path) as img:
        w, h = img.size
    n_tiles = math.ceil(w / 512) * math.ceil(h / 512) + 1
    return n_tiles * 256

def get_image_paths(media_folder: Path, media_file: str) -> list[Path]:
    paths = []
    stem = Path(media_file).name
    if Path(media_file).suffix.lower() in COHERE_TRANSFORMABLE_FORMATS:
        paths = sorted(media_folder.glob(f"{stem}__*.jpg"))
    if Path(media_file).suffix.lower() in COHERE_COMPATIBLE_FORMATS:
        jpg_path = (media_folder / stem).with_suffix(".jpg")
        if jpg_path.exists():
            paths = [jpg_path]

    logger.debug(f"Found {len(paths)} path(s) for file: {media_file}")
    return paths