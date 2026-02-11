import numpy as np
from typing import Callable, Tuple, List, Union

from enum import Enum

import torch
from torch.autograd.functional import jacobian

from geodesic_funcs import ExpMethod, LogMethod, DistSquaredMap, dist_map
from metric import MetricField, Metric, MetricView, RnMetricField
from connection import Connection

from dataclasses import dataclass

from geodesic_funcs import DistSquaredMap

from time import time


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
    damp = 0.8
    damp_growth = 0.95  # decays (helps with eventual convergence)
    max_iters = 1000


@dataclass
class SolverResult:
    success: bool
    iters: int
    p: torch.tensor


def riem_grad_descent(
    f, p0, mfld_cfg: MfldCfg, solv_cfg: SolverCfg, *args
) -> SolverResult:
    # standard optimization algorithm (for us this will act as one of the
    # available subsolvers to be used by ralm)

    p_prev = None
    p = p0

    for i in range(solv_cfg.max_iters):
        df = jacobian(lambda p: f(p, mfld_cfg, *args), p, create_graph=True)
        grad_f = mfld_cfg.metric_field(p).sharp(df)

        p = mfld_cfg.exp_method(p, -solv_cfg.damp * grad_f, mfld_cfg.conn)

        if (
            p_prev is not None
            and dist_map(
                p, p_prev, mfld_cfg.metric_field, mfld_cfg.conn, mfld_cfg.dist_method
            )
            <= solv_cfg.conv_eps
        ):
            return SolverResult(True, i + 1, p)

        p_prev = p.clone()  # otherwise it p == p_prev

        if solv_cfg.damp_growth is not None:
            solv_cfg.damp *= solv_cfg.damp_growth

    return SolverResult(False, solv_cfg.max_iters, p)


class SubsolverMethod(Enum):
    RIEM_GRAD_DESCENT = riem_grad_descent

    def __call__(self, *vars):
        self.value(*vars)


@dataclass
class ConstrainedSolverCfg:
    sub_method: SubsolverMethod
    sub_cfg: SolverCfg

    g_mult_clips: Union[List[Tuple[float, float]], Tuple[float, float]] = (
        -10.0,
        10.0,
    )  # clip on lagrangian mults of gs
    h_mult_clips: Union[List[Tuple[float, float]], Tuple[float, float]] = (
        -10.0,
        10.0,
    )  # clip on lagrangian mults of hs

    penalty = 0.8
    penalty_growth = 1.1  # grows over time
    conv_eps = 1e-6
    max_iters = 100
    eq_eps = 1e-6  # permissable abs error for equality constraints


@dataclass
class ConstrainedSolverResult:
    success: bool
    converged: bool
    constrs_violated: bool
    subsolver_failed: bool
    iters: int
    p: torch.tensor
    g_mults: List[Tuple[float, float]]  # g lagrange multipliers
    h_mults: List[Tuple[float, float]]  # h lagrange multipliers
    gs_eval: List[float]  # value of the constraints
    hs_eval: List[float]  # value of the constraints


def _ralm_subproblem(p, rho, f, gs, hs, mu_mults, lambda_mults, mfld_cfg, *args):
    sum = 0
    for i in range(len(gs)):
        sum += torch.maximum(
            torch.tensor(0.0), mu_mults[i] / rho + gs[i](p, mfld_cfg, *args)
        )
    for j in range(len(hs)):
        sum += (hs[j](p, mfld_cfg, *args) + lambda_mults[j] / rho) ** 2

    return f(p, mfld_cfg, *args) + rho / 2 * sum


def _constraints_violated(p, gs, hs, mfld_cfg: MfldCfg, eq_eps, *args):
    gs_eval = torch.tensor([g(p, mfld_cfg, *args) for g in gs])
    hs_eval = torch.tensor([h(p, mfld_cfg, *args) for h in hs])

    constr_violated = torch.any(gs_eval > 0.0) or torch.any(hs_eval.abs() > eq_eps)

    return constr_violated, gs_eval, hs_eval


def ralm(f, gs, hs, p0, mfld_cfg: MfldCfg, solve_cfg: ConstrainedSolverCfg, *args):
    if solve_cfg.sub_cfg is None:
        raise ValueError("Subsolver configuration must be provided to use RALM")

    p_prev = None
    p = p0

    n = len(gs)  # number of inequalities
    m = len(hs)  # number of equalities

    g_mults = torch.zeros((n,))
    h_mults = torch.zeros((m,))

    for i in range(solve_cfg.max_iters):
        # print(f"i: {i}")

        # finds the point that minimizes the augmented lagrangian function with
        # with the current lagrangian multipliers

        alf_result = solve_cfg.sub_method(
            lambda p, mfld_cfg, *args: _ralm_subproblem(
                p, solve_cfg.penalty, f, gs, hs, g_mults, h_mults, mfld_cfg, *args
            ),
            p,
            mfld_cfg,
            solve_cfg.sub_cfg,
            *args,
        )
        p = alf_result.p

        constr_violated, gs_eval, hs_eval = _constraints_violated(
            p, gs, hs, mfld_cfg, solve_cfg.eq_eps, *args
        )

        if not alf_result.success:
            return ConstrainedSolverResult(
                False,
                False,
                constr_violated,
                True,
                i + 1,
                p,
                g_mults,
                h_mults,
                gs_eval,
                hs_eval,
            )

        if (
            p_prev is not None
            and dist_map(
                p, p_prev, mfld_cfg.metric_field, mfld_cfg.conn, mfld_cfg.dist_method
            )
            <= solve_cfg.conv_eps
            and not constr_violated
        ):
            # if subsolver converges and constraints aren't violated then exit
            # early from the optimization process (if constraints are violated
            # then continues with iteration as the penalty will grow which
            # should hopefully allow constraint satisfaction)
            return ConstrainedSolverResult(
                True,
                True,
                False,
                False,
                i + 1,
                p,
                g_mults,
                h_mults,
                gs_eval,
                hs_eval,
            )

        # not converged so update the lagrangians then continue with with
        # attempting to optimize the augmented lagrangian function

        for j in range(n):
            gj_min_clip, gj_max_clip = (
                solve_cfg.g_mult_clips
                if type(solve_cfg.g_mult_clips) is tuple
                else solve_cfg.g_mult_clips[j]
            )
            g_mults[j] = torch.clip(
                g_mults[j] + solve_cfg.penalty * gs_eval[j], gj_min_clip, gj_max_clip
            )
        for j in range(m):
            hj_min_clip, hj_max_clip = (
                solve_cfg.h_mult_clips
                if type(solve_cfg.h_mult_clips) is tuple
                else solve_cfg.h_mult_clips[j]
            )
            h_mults[j] = torch.clip(
                h_mults[j] + solve_cfg.penalty * hs_eval[j], hj_min_clip, hj_max_clip
            )

        p_prev = p

        if solve_cfg.penalty_growth is not None:
            solve_cfg.penalty *= solve_cfg.penalty_growth

    constr_violated, gs_eval, hs_eval = _constraints_violated(
        p, gs, hs, mfld_cfg, solve_cfg.eq_eps, *args
    )
    return ConstrainedSolverResult(
        False,
        False,
        constr_violated,
        True,
        solve_cfg.max_iters,
        p,
        g_mults,
        h_mults,
        gs_eval,
        hs_eval,
    )


def test_riem_grad_descent():
    p = torch.tensor([1.0, 2.0])
    q = torch.tensor([4.0, -1.0])

    def f(p, cfg: MfldCfg, q):
        return 0.5 * dist_squared_map(p, q, cfg)

    g = RnMetricField(2)
    mfld_cfg = MfldCfg(g, g.christoffels())
    solv_cfg = SolverCfg()

    result = riem_grad_descent(f, p, mfld_cfg, solv_cfg, q)

    print(f"riem result: {result}")


def test_ralm():
    with torch.profiler.profile() as prof:
        # NOTE: from testing it seems that generally we want the penalty growth
        # to be larger than the decay rate of the subsolver (seems to sometimes
        # get stuck at a position that violates constraints if we decay too quickly)

        p = torch.tensor([-2.0, 0.0])
        q = torch.tensor([0.0, 0.0])  # want to approach this point

        c = torch.tensor([2.0, 0.0])  # circle centerd on opposite side of p
        rad = 1.0

        def f(p, cfg: MfldCfg, q, c, rad):
            return 0.5 * dist_squared_map(p, q, cfg)

        def g(p, cfg: MfldCfg, q, c, rad):
            # all g must be defined such that g(p) <= 0
            return dist_squared_map(p, c, cfg) - rad**2

        metric_field = RnMetricField(2)
        mfld_cfg = MfldCfg(metric_field, metric_field.christoffels())

        constr_solv_cfg = ConstrainedSolverCfg(
            SubsolverMethod.RIEM_GRAD_DESCENT, SolverCfg()
        )

        # TODO: improve this optimization time (likely recomputing values so can
        # likely implement caching somewhere in the architecture)

        start_time = time()
        result = ralm(f, [g], [], p, mfld_cfg, constr_solv_cfg, q, c, rad)
        end_time = time()

        print(f"ralm result: {result}")
        print(f"time: {end_time - start_time}")

    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    # test_riem_grad_descent()
    test_ralm()
