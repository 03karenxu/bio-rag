
import fitz
import math
import logging
from PIL import Image
from io import BytesIO
from pathlib import Path
from config import COHERE_MAX_W, COHERE_MAX_H, COHERE_TRANSFORMABLE_FORMATS, COHERE_COMPATIBLE_FORMATS

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

def get_image_paths(media_path: Path) -> list[Path]:
    if media_path.suffix in COHERE_TRANSFORMABLE_FORMATS:
        return sorted(media_path.parent.glob(f"{media_path.name}_p*.png"))
    if media_path.suffix in COHERE_COMPATIBLE_FORMATS and media_path.exists():
        return [media_path]
    return []

def is_oversized(img: Image.Image):
    '''
    returns true if the img (PIL Image) is too large for cohere embed v4, else false
    '''
    w, h = img.size
    return w > COHERE_MAX_W or h > COHERE_MAX_H


def fit_to_max(img: Image.Image) -> Image.Image:
    '''
    resizes an image to fit cohere embed v4 max dimensions
    '''
    if not is_oversized(img):
        return img
    w, h = img.size
    logger.info("Resizing image...")
    scale = min(COHERE_MAX_W / w, COHERE_MAX_H / h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def resize_and_save(path: Path, content: bytes) -> None:
    '''
    resizes an image from a path and saves it to disk
    '''
    img = Image.open(BytesIO(content))
    img = fit_to_max(img)
    img.save(path)


def pdf_to_png(content: bytes) -> list[bytes]:
    '''converts a pdf to a series of pngs (one per page)'''
    doc = fitz.open(stream=content, filetype="pdf")
    results = []
    for page in doc:
        with Image.open(BytesIO(page.get_pixmap(dpi=150).tobytes("png"))) as img:
            img = fit_to_max(img)
            buf = BytesIO()
            img.save(buf, format="PNG")
        results.append(buf.getvalue())
    return results


def tif_to_png(content: bytes) -> bytes:
    '''also works for tiff'''
    with Image.open(BytesIO(content)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = fit_to_max(img)
        buf = BytesIO()
        img.save(buf, format="PNG")
    return buf.getvalue()


def save_as_png(filename: str, content: bytes, out_dir: Path) -> None:
    '''
    converts a file to PNG(s) and saves to disk.
    '''
    path = out_dir / filename
    suffix = Path(filename).suffix.lower()

    logger.info(f"Converting {filename} to .png...")

    if suffix == ".pdf":
        png_bytes = pdf_to_png(content)
    elif suffix in (".tif", ".tiff"):
        png_bytes = [tif_to_png(content)]
    else:
        raise AttributeError(f"Unsupported format {suffix}")

    for i, b in enumerate(png_bytes):
        path.with_stem(f"{path.name}__{i}").with_suffix(".png").write_bytes(b)