import torch

from abc import ABC, abstractmethod

from diff_mfld_optim.mfld_util import MfldCfg
from diff_mfld_optim.geometry.connection import Connection

from funcs import FuncArgs


class ProdMfldFunc(ABC):
    @abstractmethod
    def value(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        optim_cfg: MfldCfg,
        prod_conn: Connection,
        *func_args: *FuncArgs,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def grad(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        optim_cfg: MfldCfg,
        prod_conn: Connection,
        *func_args: *FuncArgs,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def hess(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        optim_cfg: MfldCfg,
        prod_conn: Connection,
        *func_args: *FuncArgs,
    ) -> torch.Tensor:
        pass

    def __rmul__(self, other):
        # assume that the types make sense here
        value = lambda p, q, cfg, *args: other * self.value(p, q, cfg, *args)
        diff = lambda p, q, cfg, *args: other * self.diff(p, q, cfg, *args)
        hess = lambda p, q, cfg, *args: other * self.hess(p, q, cfg, *args)

        return ConstrProdMfldFunc(value, diff, hess)

    def __mul__(self, other):
        return other * self


class ConstrProdMfldFunc(ProdMfldFunc):
    def __init__(self, value, diff, hess):
        # uses the provided lambda functions as the manifold function values
        self.value = value
        self.diff = diff
        self.hess = hess

class ProductSquaredDist(ProdMfldFunc):
    # implements an analogous "distance" function on the product manifold whose
    # approximate value is the riemannian distance on the optimization manifold
    # as used before and so the differentia
    
    def value(self, p, q, optim_cfg, prod_conn, *func_args):


        return super().value(p, q, optim_cfg, prod_conn, *func_args)