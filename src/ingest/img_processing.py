import fitz
import logging
from PIL import Image
from io import BytesIO
from pathlib import Path
from config import COHERE_MAX_W, COHERE_MAX_H

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

def fit_to_bounds(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img
    scale = min(max_w / w, max_h / h)
    logger.debug(f"Resizing image from {w}x{h}...")
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def resize_and_save(path: Path, content: bytes, max_w: int = COHERE_MAX_W, max_h: int = COHERE_MAX_H) -> None:
    img = Image.open(BytesIO(content))
    img = fit_to_bounds(img, max_w, max_h)
    img.convert("RGB").save(path.with_suffix(".jpg"), format="JPEG", quality=85)


def pdf_to_jpg(content: bytes, max_w: int = COHERE_MAX_W, max_h: int = COHERE_MAX_H) -> list[bytes]:
    results = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            with Image.open(BytesIO(page.get_pixmap(dpi=150).tobytes("png"))) as img:
                img = fit_to_bounds(img, max_w, max_h)
                buf = BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
            results.append(buf.getvalue())
    return results


def tif_to_jpg(content: bytes, max_w: int = COHERE_MAX_W, max_h: int = COHERE_MAX_H) -> bytes:
    with Image.open(BytesIO(content)) as img:
        img = fit_to_bounds(img.convert("RGB"), max_w, max_h)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def save_as_jpg(outpath: Path, content: bytes, max_w: int = COHERE_MAX_W, max_h: int = COHERE_MAX_H) -> None:
    suffix = outpath.suffix.lower()
    logger.debug(f"Converting {outpath.name} to .jpg...")

    if suffix == ".pdf":
        jpg_bytes = pdf_to_jpg(content, max_w, max_h)
    elif suffix in (".tif", ".tiff"):
        jpg_bytes = [tif_to_jpg(content, max_w, max_h)]
    else:
        raise AttributeError(f"Unsupported format {suffix}")

    for i, b in enumerate(jpg_bytes):
        outpath.with_stem(f"{outpath.stem}__{i}").with_suffix(".jpg").write_bytes(b)