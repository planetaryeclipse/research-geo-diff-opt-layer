from enum import Enum

import torch

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.results import SubsolverCfg
from optim.subsolvers.rgd import riem_grad_descent, RiemGradDescentCfg
from optim.subsolvers.rtr import riem_trust_region, RiemTrustRegionCfg


class SubsolverMethod(Enum):
    RIEM_GRAD_DESCENT = (riem_grad_descent, RiemGradDescentCfg)
    RIEM_TRUST_REGION = (riem_trust_region, RiemTrustRegionCfg)

    def __call__(self, f: MfldFunc, p0: torch.Tensor, mfld: ComputeMfld, cfg: SubsolverCfg,
                 *args: *FuncArgs):
        method, cfg_type = self.value

        if not isinstance(cfg, cfg_type):
            raise ValueError(f"configuration of type {type(cfg)} cannot be used with subsolver method {self.name}")

        result = method(f, p0, mfld, cfg, *args)  # calls the method (with having checked that the correct cfg is given)
        # return result.result  # converts from custom return type to the main return type
        return result  # returns the regular return type
