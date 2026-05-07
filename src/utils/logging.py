import logging

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    disable_import_logging()
    
def disable_import_logging():
    for name in ["litellm", "httpx", "httpcore", "boto3", "botocore", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)