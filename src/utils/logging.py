import logging
from config import LOG_DIR

def init_logging(log_file: str | None = None):
    if log_file:
        log_file = LOG_DIR / log_file
    handler = logging.FileHandler(log_file, mode="w") if log_file else logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.handlers.clear()
    root.addHandler(handler)
    
    for name in ("__main__", "utils.embeddings", "utils.paper_parser", "utils.image_processing"):
        logging.getLogger(name).setLevel(logging.INFO)
    
    # logging.getLogger("__main__").setLevel(logging.DEBUG)