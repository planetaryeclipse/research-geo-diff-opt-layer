from typing import List

import torch

from enum import Enum

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.constrained.ralm import RalmCfg, ralm
from optim.constrained.repm import RepmCfg, repm
from optim.constrained.rsqo import rsqo, RsqoCfg
from optim.results import ConstrSolverCfg


class ConstrSolverMethod(Enum):
    RALM = (ralm, RalmCfg)
    REPM = (repm, RepmCfg)
    RSQO = (rsqo, RsqoCfg)

    def __call__(self, f: MfldFunc, gs: List[MfldFunc], hs: List[MfldFunc], p0: torch.Tensor, mfld: ComputeMfld,
                 cfg: ConstrSolverCfg, *args: *FuncArgs):
        method, cfg_type = self.value

        if not isinstance(cfg, cfg_type):
            raise ValueError(
                "configuration of type {type(cfg)} cannot be used with constrained solver method {self.name}")

        result = method(f, gs, hs, p0, mfld, cfg, *args)
        return result  # returns the regular return type
