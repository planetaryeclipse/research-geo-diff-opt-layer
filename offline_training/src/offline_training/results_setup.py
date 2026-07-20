from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path


def get_run_dir(results_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / f"run_{timestamp}"
    run_dir.mkdir()

    return run_dir
