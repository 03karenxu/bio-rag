import logging

def init_logging(log_file: str | None = None):
    handler = logging.FileHandler(log_file, mode="w") if log_file else logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    
    disable_import_logging()
    
def disable_import_logging():
    for name in logging.root.manager.loggerDict:
        if name not in ("__main__", "utils.embed", "utils.xml_parser", "utils.image_handling"):
            logging.getLogger(name).setLevel(logging.WARNING)