from dataclasses import dataclass

@dataclass
class FetchedPaper:
    identifier: str
    main_file: tuple[str, bytes] # (filename, raw bytes)
    media_files: dict[str, bytes] | None = None # {filename: raw bytes}