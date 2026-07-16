from pathlib import Path
from typing import List


def clean_dirs(dirs: List[Path]):
    for dir in dirs:
        for file in dir.iterdir():
            if file.name == ".gitkeep":
                continue
            file.unlink()
