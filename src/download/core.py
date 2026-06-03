import logging
from pathlib import Path
from dataclasses import dataclass
from config import COHERE_COMPATIBLE_FORMATS
from download.img_processing import save_as_jpg, resize_and_save

logger = logging.getLogger(__name__)

@dataclass
class FetchedPaper:
    identifier: str
    main_file: tuple[str, bytes]
    media_files: dict[str, bytes] | None = None

class Downloader():
    '''
    downloads a FetchedPaper
    '''
    def download_paper(self, paper: FetchedPaper, out_dir: Path, wrap_main_file: bool = False) -> None:
        # download main paper file
        paper_root = out_dir / "paper" if wrap_main_file else out_dir
        main_filename, main_bytes = paper.main_file
        paper_root.mkdir(parents=True, exist_ok=True)
        paper_path = paper_root / main_filename

        with open(paper_path, "wb") as f:
            f.write(main_bytes)
        
        logger.info(f"Downloaded paper for {paper.identifier} to {paper_path}")

        # download media files
        if paper.media_files:
            media_dir = out_dir / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in paper.media_files.items():
                outpath = media_dir / filename
                if Path(filename).suffix.lower() in COHERE_COMPATIBLE_FORMATS:
                    resize_and_save(outpath, content)
                elif Path(filename).suffix:
                    try:
                        save_as_jpg(outpath, content)
                    except Exception as e:
                        logger.warning(f"Could not convert {filename} to .jpg, saving as-is: {e}")
                        outpath.write_bytes(content)

            logger.info(f"Downloaded media files for {paper.identifier} to {media_dir}")