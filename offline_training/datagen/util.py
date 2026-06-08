import itertools

from pathlib import Path
from typing import List

EPISODES_DIR = Path(__file__).parent.joinpath("../data/episodes")
EPISODE_FILE_PREFIX = "episode_{id:003d}.npz"

CONTROLS_DIR = Path(__file__).parent.joinpath("../data/controls")
CONTROLS_FILE_PREFIX = "controls_{id:003d}.npz"

_PARENT_INSTANCES_DIR = Path(__file__).parent.joinpath("../data/instances")
INDIV_INSTANCE_DIR = _PARENT_INSTANCES_DIR / "individual"
INDIV_INSTANCE_FILE_PREFIX = "instance_{id:003d}.npz"

TRAIN_INSTANCE_PATH = _PARENT_INSTANCES_DIR / "training.npz"
VALID_INSTANCE_PATH = _PARENT_INSTANCES_DIR / "validation.npz"


def clean_dirs(dirs: List[Path]):
    for dir in dirs:
        for file in dir.iterdir():
            if file.name == ".gitkeep":
                continue
            file.unlink()
