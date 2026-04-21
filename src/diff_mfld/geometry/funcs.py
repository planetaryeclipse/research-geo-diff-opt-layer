import torch

from typing import Callable, TypeVarTuple, List, Tuple, Any
from dataclasses import dataclass

from src.diff_mfld.mfld import Mfld, ComputeMfld

from abc import ABC, abstractmethod

FuncArgs = TypeVarTuple("FuncArgs")


class MfldFunc(ABC):
    @abstractmethod
    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        pass

    @abstractmethod
    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        pass

    @abstractmethod
    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        pass

    def __rmul__(self, other):
        # assume that the types make sense here
        value = lambda p, cfg, *args: other * self.value(p, cfg, *args)
        diff = lambda p, cfg, *args: other * self.diff(p, cfg, *args)
        hess = lambda p, cfg, *args: other * self.hess(p, cfg, *args)

        return ConstructedMfldFunc(value, diff, hess)

    def __mul__(self, other):
        return other * self


class ConstructedMfldFunc(MfldFunc):
    def __init__(self, value, diff, hess):
        # uses the provided lambda fucntions as the manifold function values
        self._value = value
        self._diff = diff
        self._hess = hess

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._value(p, cfg, *args)

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._diff(p, cfg, *args)

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        return self._hess(p, cfg, *args)


class RiemannSquaredDist(MfldFunc):
    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args):
        q: torch.Tensor = args[0]

        g = cfg.mfld.metric(p)
        v = cfg.dist_method(p, q, cfg.mfld.conn)
        dist_sqr = g(v, v)  # inner product of v with itself under metric g

        return dist_sqr

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args):
        q: torch.Tensor = args[0]

        g = cfg.mfld.metric(p)
        v = cfg.dist_method(p, q, cfg.mfld.conn)
        dv = g.flat(v)  # differential in cotangent space

        diff = -2 * dv
        return diff

    def hess(self, p, cfg: ComputeMfld, *args):
        q: torch.Tensor = args[0]

        g = cfg.mfld.metric(p).mat
        conn_coeffs = cfg.mfld.conn(p)

        v = cfg.dist_method(p, q, cfg.mfld.conn)

        # index of basis of diff is in the last dimension
        metric_partials = cfg.mfld.metric.partials(p)

        cov_hess = torch.tensordot(metric_partials, v, ([1], [0]))
        cov_hess += -g
        cov_hess += -torch.tensordot(
            torch.tensordot(g, v, ([1], [0])), conn_coeffs, ([0], [0])
        )
        cov_hess *= -2.

        return cov_hess
