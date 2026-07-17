import cattrs
import numpy as np

from pathlib import Path
from typing import List


def clean_dirs(dirs: List[Path]):
    for dir in dirs:
        for file in dir.iterdir():
            if file.name == ".gitkeep":
                continue
            file.unlink()


def _flatten(d: dict, prefix: str, out: dict):
    """Recursively flatten a nested dict (as produced by cattrs.unstructure)
    into flat '/'-joined keys, e.g. {'trajectory': {'t': arr}} -> {'trajectory/t': arr}
    """
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, key + "/", out)
        else:
            out[key] = np.asarray(v)


def _unflatten(flat: dict) -> dict:
    """Rebuild a nested dict from flat '/'-joined keys."""
    nested = {}
    for key, value in flat.items():
        parts = key.split("/")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return nested


_np_converter = cattrs.Converter()
_np_converter.register_unstructure_hook(np.ndarray, lambda a: a.tolist())
_np_converter.register_structure_hook(np.ndarray, lambda v, _: np.array(v))


class Serializable:
    def save(self, path: str | Path):
        flat_data = {}
        _flatten(_np_converter.unstructure(self), "", flat_data)
        np.savez(path, **flat_data, allow_pickle=True)

    @classmethod
    def load(cls, path: str | Path):
        data = _unflatten(np.load(path, allow_pickle=True))
        return _np_converter.structure(data, cls)
