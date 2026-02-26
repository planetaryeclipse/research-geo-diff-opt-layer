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
    FuncArgs,
)
from diff_mfld_optim.geodesic.geodesic_funcs import dist_map
from diff_mfld_optim.mfld_util import MfldCfg

from diff_mfld_optim.geometry.funcs import (
    MfldFunc,
    FuncArgs,
)


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


def _constraints_violated(
    p: torch.Tensor,
    gs: List[MfldFunc],
    hs: List[MfldCfg],
    cfg: MfldCfg,
    eq_eps: float,
    *args: *FuncArgs,
):
    g_vals = torch.tensor([g.value(p, cfg, *args) for g in gs])
    h_vals = torch.tensor([h.value(p, cfg, *args) for h in hs])

    return (
        torch.any(g_vals > eq_eps) or torch.any(h_vals.abs() > eq_eps),
        g_vals,
        h_vals,
    )


class AugmentedLagrangian(MfldFunc):
    def __init__(
        self,
        f: MfldFunc,
        gs: List[MfldFunc],
        hs: List[MfldFunc],
        penalty: float,
        g_mults: List[float],
        h_mults: List[float],
    ):
        self._f = f
        self._gs = gs
        self._hs = hs

        self._penalty = penalty
        self._g_mults = g_mults
        self._h_mults = h_mults

    def value(self, p, cfg, *args):
        aug_value = self._f.value(p, cfg, *args)

        aug_sum = 0
        aug_sum += sum(
            torch.maximum(0, g_mult / self._penalty + g.value(p, cfg, *args) ** 2)
            for g, g_mult in zip(self._gs, self._g_mults)
        )
        aug_sum += sum(
            (h.value(p, cfg, *args) + h_mult / self._penalty) ** 2
            for h, h_mult in zip(self._hs, self._h_mults)
        )
        aug_value += self._penalty / 2 * aug_sum

        return aug_value

    def diff(self, p, cfg, *args):
        aug_diff = self._f.diff(p, cfg, *args)

        g_vals = [g.value(p, cfg, *args) for g in self._gs]

        aug_sum = 0
        aug_sum += sum(
            (
                (g_val + g_mult / self._penalty) * g.diff(p, cfg, *args)
                if g_val > 0.0
                else 0.0
            )
            for g, g_mult, g_val in zip(self._gs, self._g_mults, g_vals)
        )
        aug_sum += sum(
            (h.value(p, cfg, *args) + h_mult / self._penalty) * h.diff(p, cfg, *args)
            for h, h_mult in zip(self._hs, self._h_mults)
        )
        aug_diff += self._penalty * aug_sum

        return aug_diff

    def hess(self, p, cfg, *args):
        aug_hess = self._f.hess(p, cfg, *args)

        g_vals = [g.value(p, cfg, *args) for g in self._gs]
        g_diffs = [g.diff(p, cfg, *args) for g in self._gs]
        h_diffs = [h.diff(p, cfg, *args) for h in self._hs]

        aug_sum = 0
        aug_sum += sum(
            (
                torch.outer(g_diff, g_diff)
                + (g_val + g_mult / self._penalty) * g.hess(p, cfg, *args)
                if g_val > 0.0
                else 0.0
            )
            for g, g_mult, g_val, g_diff in zip(
                self._gs, self._g_mults, g_vals, g_diffs
            )
        )
        ang_sum += sum(
            torch.outer(h_diff, h_diff)
            + (h.value(p, cfg, *args) + h_mult / self._penalty)
            for h, h_mult, h_diff in zip(self._hs, self._h_mults, h_diffs)
        )
        aug_hess += self._penalty * ang_sum

        return aug_hess

    @property
    def penalty(self):
        return self._penalty

    @property
    def g_mults(self):
        return self._g_mults

    @property
    def h_mults(self):
        return self._h_mults

    @penalty.setter
    def penalty(self, penalty):
        self._penalty = penalty

    @g_mults.setter
    def g_mults(self, g_mults):
        self._g_mults = g_mults

    @h_mults.setter
    def h_mults(self, h_mults):
        self._h_mults = h_mults


def ralm(
    f: MfldFunc,
    gs: List[MfldFunc],
    hs: List[MfldFunc],
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

    # setup the augmented lagrangian function which will be minimized by the
    # selected unconstrained subsolver optimization method
    aug_lagr = AugmentedLagrangian(f, gs, hs, solve_cfg.penalty, g_mults, h_mults)

    for i in range(solve_cfg.max_iters):
        # solves the optimal point of the unconstrained lagrangian function
        # which acts as an estimator of the solution of the true problem

        alf_result: SolverResult = solve_cfg.sub_method(
            aug_lagr,
            p,
            mfld_cfg,
            solve_cfg.sub_cfg,
            *args,
        )
        p = alf_result.p

        constr_violated, gs_eval, hs_eval = _constraints_violated(
            p, gs, hs, mfld_cfg, solve_cfg.eq_eps, *args
        )

        # print(
        #     f"ralm: i={i}, p={p}, penalty={solve_cfg.penalty}, "
        #     f"g_mults={g_mults}, g_vals={gs_eval}, "
        #     f"h_mults={h_mults}, h_vals={hs_eval}"
        # )

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
        elif (
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

        # not converged so update the multipliers then continue with with
        # attempting to optimize the augmented lagrangian function

        unpack_clip_bounds = lambda clips, j: (
            clips if type(clips) is tuple else clips[j]
        )
        g_mults = torch.tensor(
            [
                torch.clip(
                    g_mults[j] + solve_cfg.penalty * gs_eval[j],
                    *unpack_clip_bounds(solve_cfg.g_mult_clips, j),
                )
                for j in range(n)
            ]
        )
        h_mults = torch.tensor(
            [
                torch.clip(
                    h_mults[j] + solve_cfg.penalty * hs_eval[j],
                    *unpack_clip_bounds(solve_cfg.h_mult_clips, j),
                )
                for j in range(m)
            ]
        )

        # updates the penalty
        sigma = np.maximum(gs_eval.numpy(), -g_mults.numpy() / solve_cfg.penalty)
        if len(hs_eval) > 0:
            arr_max = np.max(
                np.concat(np.abs(hs_eval.numpy()).flatten()),
                np.abs(sigma).flatten(),
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

        # applies the changes to the augmetned lagrangian for use in the next
        # attempt at solving the constrained optimziation problem
        aug_lagr.penalty = solve_cfg.penalty
        aug_lagr.g_mults = g_mults
        aug_lagr.h_mults = h_mults

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
        f: MfldFunc,
        gs: List[MfldFunc],
        hs: List[MfldFunc],
        p0: torch.Tensor,
        mfld_cfg: MfldCfg,
        solve_cfg: ConstrainedSolverCfg,
        *func_args: *FuncArgs,
    ):
        self.value(f, gs, hs, p0, mfld_cfg, solve_cfg, *func_args)
