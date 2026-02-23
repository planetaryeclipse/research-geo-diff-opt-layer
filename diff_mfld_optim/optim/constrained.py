import torch
import numpy as np

from copy import deepcopy, copy
from enum import Enum
from dataclasses import dataclass
from typing import Union, List, Tuple

from diff_mfld_optim.optim.subsolver import (
    SubsolverMethod,
    SolverCfg,
    SolverResult,
    OptimFunc,
    FuncArgs,
)
from diff_mfld_optim.geodesic.geodesic_funcs import dist_map
from diff_mfld_optim.mfld_util import MfldCfg


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

    ratio = 0.6

    conv_eps = 1e-6
    max_iters = 100
    eq_eps = 1e-6  # permissable abs error for equality constraints

    subsolver_acc = 0.1
    subsolver_acc_min = 1e-3
    subsolver_acc_growth = 0.9


@dataclass
class ConstrainedSolverResult:
    success: bool
    converged: bool
    constrs_violated: bool
    subsolver_failed: bool
    iters: int
    p: torch.Tensor
    p0: torch.Tensor
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
    gs_eval = torch.tensor([g(p.detach(), mfld_cfg, *args) for g in gs])
    hs_eval = torch.tensor([h(p.detach(), mfld_cfg, *args) for h in hs])

    constr_violated = torch.any(gs_eval > 0.0) or torch.any(hs_eval.abs() > eq_eps)

    return constr_violated, gs_eval, hs_eval


def ralm(
    f: OptimFunc,
    gs: List[OptimFunc],
    hs: List[OptimFunc],
    p0: torch.Tensor,
    mfld_cfg: MfldCfg,
    solve_cfg: ConstrainedSolverCfg,
    *args: *FuncArgs,
):
    if solve_cfg.sub_cfg is None:
        raise ValueError("Subsolver configuration must be provided to use RALM")

    # clones the constrained solver configuration values (not the subsolver method
    # which causes issues) so each run of ralm is the same
    solve_cfg = copy(solve_cfg)

    p_prev = None
    p = p0.detach().clone()

    n = len(gs)  # number of inequalities
    m = len(hs)  # number of equalities

    g_mults = torch.zeros((n,))
    h_mults = torch.zeros((m,))

    # force the constrained solver accuracy properties on the subsolver cfg
    solve_cfg.sub_cfg.conv_eps = solve_cfg.subsolver_acc

    # print(f"Starting constrained!")

    for i in range(solve_cfg.max_iters):
        # print(f"ralm: i={i}")

        # finds the point that minimizes the augmented lagrangian function with
        # with the current lagrangian multipliers

        alf_result: SolverResult = solve_cfg.sub_method(
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
            # print("Sub solver failed!")
            return ConstrainedSolverResult(
                False,
                False,
                constr_violated,
                True,
                i + 1,
                p,
                p0,
                g_mults,
                h_mults,
                gs_eval,
                hs_eval,
            )

        if (
            p_prev is not None
            and dist_map(
                p,
                p_prev,
                mfld_cfg.metric_field,
                mfld_cfg.conn,
                mfld_cfg.dist_method,
            )
            <= solve_cfg.conv_eps
            and not constr_violated
        ):
            # if subsolver converges and constraints aren't violated then exit
            # early from the optimization process (if constraints are violated
            # then continues with iteration as the penalty will grow which
            # should hopefully allow constraint satisfaction)
            # print("Ended successfully!")
            return ConstrainedSolverResult(
                True,
                True,
                False,
                False,
                i + 1,
                p,
                p0,
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
                g_mults[j] + solve_cfg.penalty * gs_eval[j],
                gj_min_clip,
                gj_max_clip,
            )
        for j in range(m):
            hj_min_clip, hj_max_clip = (
                solve_cfg.h_mult_clips
                if type(solve_cfg.h_mult_clips) is tuple
                else solve_cfg.h_mult_clips[j]
            )
            h_mults[j] = torch.clip(
                h_mults[j] + solve_cfg.penalty * hs_eval[j],
                hj_min_clip,
                hj_max_clip,
            )

        # updates the penalty
        sigma = np.maximum(gs_eval.numpy(), -g_mults.numpy() / solve_cfg.penalty)
        if len(hs_eval) > 0:
            arr_max = np.max(
                np.concat(np.abs(hs_eval.numpy()).flatten()), np.abs(sigma).flatten()
            )
        else:
            arr_max = np.max(np.abs(sigma).flatten())

        if not (i == 0 or arr_max <= solve_cfg.ratio * arr_max):
            solve_cfg.penalty *= solve_cfg.penalty_growth

        # updates the subsolver accuracy
        solve_cfg.sub_cfg.conv_eps = np.maximum(
            solve_cfg.sub_cfg.conv_eps * solve_cfg.subsolver_acc_growth,
            solve_cfg.subsolver_acc_min,
        )

        # preps for the next round
        p_prev = p

    # print("Iters exceeded!")

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
        p0,
        g_mults,
        h_mults,
        gs_eval,
        hs_eval,
    )


class ConstrainedSolverMethod(Enum):
    RALM = ralm

    def __call__(
        self,
        f: OptimFunc,
        gs: List[OptimFunc],
        hs: List[OptimFunc],
        p0: torch.Tensor,
        mfld_cfg: MfldCfg,
        solve_cfg: ConstrainedSolverCfg,
        *func_args: *FuncArgs,
    ):
        self.value(f, gs, hs, p0, mfld_cfg, solve_cfg, *func_args)
