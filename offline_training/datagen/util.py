import itertools

from pathlib import Path
from typing import List

EPISODES_TRAIN_DIR = Path(__file__).parent.joinpath("../data/episodes/train")
EPISODES_VALID_DIR = Path(__file__).parent.joinpath("../data/episodes/test")

MPC_TRAIN_DIR = Path(__file__).parent.joinpath("../data/nominal_mpc/train")
MPC_VALID_DIR = Path(__file__).parent.joinpath("../data/nominal_mpc/test")

KO_TRAIN_DIR = Path(__file__).parent.joinpath("../data/keep_out/train")
KO_VALID_DIR = Path(__file__).parent.joinpath("../data/keep_out/test")


def clean_dirs(dirs: List[Path]):
    for dir in dirs:
        for file in dir.iterdir():
            if file.name == ".gitkeep":
                continue
            file.unlink()
