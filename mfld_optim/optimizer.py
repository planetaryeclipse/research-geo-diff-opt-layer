import numpy as np
from typing import Callable, Tuple

from enum import Enum

import torch
from torch.autograd.functional import jacobian

from geodesic_funcs import ExpMethod, LogMethod, DistSquaredMap, dist_map
from metric import MetricField, Metric, MetricView, RnMetricField
from connection import Connection

from dataclasses import dataclass

from geodesic_funcs import DistSquaredMap


@dataclass
class MfldCfg:
    metric_field: MetricField
    conn: Connection

    exp_method: ExpMethod = ExpMethod.APPROX_SO
    log_method: LogMethod = LogMethod.APPROX_SO
    dist_method: LogMethod = LogMethod.APPROX_SO


def dist_squared_map(p, q, cfg: MfldCfg):
    # distance wrapper function to use clean config interface
    return DistSquaredMap.apply(p, q, cfg.metric_field, cfg.conn, cfg.dist_method)


@dataclass
class SolverCfg:
    conv_eps = 1e-6
    damp = 1.0
    max_iters = 100
    sub_cfg = None


@dataclass
class SolverResult:
    success: bool
    iters: int
    p: torch.tensor
    sub_result = None


def riemannian_grad_descent(f, p0, mfld_cfg: MfldCfg, solv_cfg: SolverCfg, *args):
    # standard optimization algorithm (for us this will act as one of the
    # available subsolvers to be used by ralm)

    success = False

    p_prev = None
    p = p0

    iter_counter = 0
    for _ in range(solv_cfg.max_iters):
        iter_counter += 1

        df = jacobian(lambda p: f(p, mfld_cfg, *args), p, create_graph=True)
        grad_f = mfld_cfg.metric_field(p).sharp(df)

        p -= solv_cfg.damp * grad_f
        if (
            p_prev is not None
            and dist_map(
                p, p_prev, mfld_cfg.metric_field, mfld_cfg.conn, mfld_cfg.dist_method
            )
            <= solv_cfg.conv_eps
        ):
            success = True
            break

        p_prev = p.clone()  # otherwise it p == p_prev

    return SolverResult(success, iter_counter, p)


def _ralm_mfld_optim():
    pass


class MfldOptimMethod(Enum):
    RALM = _ralm_mfld_optim


# accepts a position on the manifold then outputs the value and differential
Optim_Fn = Callable[
    [
        torch.tensor,
    ],
    Tuple[np.ndarray, np.ndarray],
]


@dataclass
class MfldOptimCfg:
    pass


def test():
    p = torch.tensor([1.0, 2.0])
    q = torch.tensor([4.0, -1.0])

    def f(p, cfg: MfldCfg, q):
        return 0.5 * dist_squared_map(p, q, cfg)

    g = RnMetricField(2)
    mfld_cfg = MfldCfg(g, g.christoffels())
    sub_cfg = SolverCfg()

    p_optimal = riemannian_grad_descent(f, p, mfld_cfg, sub_cfg, q)

    print(f"p_optimal: {p_optimal}")


if __name__ == "__main__":
    test()
