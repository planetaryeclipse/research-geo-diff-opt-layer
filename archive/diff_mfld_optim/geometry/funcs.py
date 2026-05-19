import torch

from typing import Callable, TypeVarTuple, List, Tuple, Any
from dataclasses import dataclass

from diff_mfld_optim.mfld_util import MfldCfg


from abc import ABC, abstractmethod

FuncArgs = TypeVarTuple("FuncArgs")


class MfldFunc(ABC):
    @abstractmethod
    def value(self, p: torch.Tensor, cfg: MfldCfg, *args: *FuncArgs) -> torch.Tensor:
        pass

    @abstractmethod
    def diff(self, p: torch.Tensor, cfg: MfldCfg, *args: *FuncArgs) -> torch.Tensor:
        pass

    @abstractmethod
    def hess(self, p: torch.Tensor, cfg: MfldCfg, *args: *FuncArgs) -> torch.Tensor:
        pass

    def __rmul__(self, other):
        # assume that the types make sense here
        value = lambda p, cfg, *args: other * self.value(p, cfg, *args)
        diff = lambda p, cfg, *args: other * self.diff(p, cfg, *args)
        hess = lambda p, cfg, *args: other * self.hess(p, cfg, *args)

        return ConstrMfldFunc(value, diff, hess)

    def __mul__(self, other):
        return other * self


class ConstrMfldFunc(ABC):
    def __init__(self, value, diff, hess):
        # uses the provided lambda fucntions as the manifold function values
        self.value = value
        self.diff = diff
        self.hess = hess


class RiemannSquaredDist(MfldFunc):
    def value(self, p, cfg: MfldCfg, q, *_args):
        g = cfg.metric_field(p)
        v = cfg.dist_method(p, q, cfg.conn)

        return g(v, v) ** 2

    def diff(self, p, cfg: MfldCfg, q, *_args):
        g = cfg.metric_field(p)
        v = cfg.dist_method(p, q, cfg.conn)
        dv = g.flat(v)  # differential in cotangent space

        diff = -2 * dv
        return diff

    def hess(self, p, cfg: MfldCfg, q, *_args):
        g = cfg.metric_field(p).mat
        conn_coeffs = cfg.conn(p)

        v = cfg.dist_method(p, q, cfg.conn)

        # index of basis of diff is in the last dimension
        metric_partials = cfg.metric_field.partials(p)

        term_1 = torch.tensordot(metric_partials, v, ([1], [0]))
        term_2 = -g
        term_3 = -torch.tensordot(
            torch.tensordot(g, v, ([1], [0])), conn_coeffs, ([0], [0])
        )

        cov_hess = -2 * (term_1 + term_2 + term_3)
        return cov_hess
