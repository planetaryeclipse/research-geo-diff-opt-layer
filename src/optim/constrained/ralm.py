import torch

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Union, Optional

from optim.results import CustomConstrSolverResult, ConstrSolverResult, ConstrSolverHistory, SubsolverCfg, \
    ConstrSolverCfg
from optim.subsolver_methods import SubsolverMethod
from optim.subsolvers.rgd import RiemGradDescentCfg
from src.diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from src.diff_mfld.mfld import Mfld, ComputeMfld


# implements the Riemannian Augmented Lagrangian Method (RALM) described in "Simple Algorithms for Optimization on
# Riemannian Manifolds with Constraints"

@dataclass
class RalmCfg(ConstrSolverCfg):
    acc_tol_min: float = 1e-3
    acc_tol_0: float = 1e-2
    acc_decay: float = 0.9  # in (0, 1)

    penalty_0: float = 0.01
    penalty_growth: float = 1.02  # > 1

    g_mult_0: float = 0.
    g_mult_min: float = 0.
    g_mult_max: float = 1000.

    h_mult_0: float = 0.
    h_mult_min: float = -1000.
    h_mult_max: float = 1000.

    ratio: float = 0.5  # in (0, 1)
    min_step: float = 0.02

    subsolver_method: SubsolverMethod = SubsolverMethod.RIEM_GRAD_DESCENT
    subsolver_cfg: SubsolverCfg = field(
        default_factory=RiemGradDescentCfg)  # must be type corresponding to the subsolver method
    max_iters: int = 1000


@dataclass
class RalmHistory:
    p_hist: torch.Tensor

    f_hist: torch.Tensor
    gs_hist: torch.Tensor
    hs_hist: torch.Tensor

    acc_hist: torch.Tensor
    penalty_hist: torch.Tensor

    g_mults_hist: torch.Tensor
    h_mults_hist: torch.Tensor

    subsolver_iters_hist: List[int]


@dataclass
class RalmResult(CustomConstrSolverResult):
    success: bool
    p: torch.Tensor
    iters: int
    history: RalmHistory

    @property
    def result(self) -> ConstrSolverResult:
        return ConstrSolverResult(
            success=self.success,
            p=self.p,
            iters=self.iters,
            history=ConstrSolverHistory(
                p_hist=self.history.p_hist,
                f_hist=self.history.f_hist,
                gs_hist=self.history.gs_hist,
                hs_hist=self.history.hs_hist,
                g_mults_hist=self.history.g_mults_hist,
                h_mults_hist=self.history.h_mults_hist,
            )
        )


class AugmentedLagrangian(MfldFunc):
    def __init__(self,
                 f: MfldFunc,
                 gs: List[MfldFunc],
                 hs: List[MfldFunc],
                 g_mults: List[float],
                 h_mults: List[float],
                 penalty: float):
        self._f = f
        self._gs = gs
        self._hs = hs
        self._g_mults = g_mults
        self._h_mults = h_mults
        self._penalty = penalty

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        g_sum = sum(
            torch.maximum(torch.tensor(0.), g_mult / self._penalty + g.value(p, cfg, *args)) ** 2
            for g, g_mult in zip(self._gs, self._g_mults)
        )
        h_sum = sum(
            (h.value(p, cfg, *args) + h_mult / self._penalty) ** 2
            for h, h_mult in zip(self._hs, self._h_mults)
        )
        aug_value = self._f.value(p, cfg, *args) + self._penalty / 2.0 * (g_sum + h_sum)
        return aug_value

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        g_vals = [g.value(p, cfg, *args) for g in self._gs]

        g_diff_sum = torch.zeros_like(p)
        g_diff_sum += sum(
            2. * (g_mult / self._penalty + g_val) * g.diff(p, cfg,
                                                           *args) if g_val + g_mult / self._penalty >= 0. else 0.0
            for g, g_mult, g_val in zip(self._gs, self._g_mults, g_vals)
        )

        h_diff_sum = torch.zeros_like(p)
        h_diff_sum += sum(
            2.0 * (h_mult / self._penalty + h.value(p, cfg, *args)) * h.diff(p, cfg, *args)
            for h, h_mult in zip(self._hs, self._h_mults)
        )
        aug_diff_value = self._f.diff(p, cfg, *args) + self._penalty / 2. * (g_diff_sum + h_diff_sum)

        # print(f"p: {p}")
        # print(f"g diff sum: {g_diff_sum}")
        # print(f"aug_lagr diff: {aug_diff_value}")

        return aug_diff_value

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        g_vals = [g.value(p, cfg, *args) for g in self._gs]
        g_diff_vals = [g.diff(p, cfg, *args) for g in self._gs]

        h_vals = [h.value(p, cfg, *args) for h in self._hs]
        h_diff_vals = [h.diff(p, cfg, *args) for h in self._hs]

        g_hess_sum = sum(
            2.0 * torch.outer(g_diff, g_diff) + 2.0 * (g_val + g_mult / self._penalty) * g.hess(p, cfg, *args)
            if g_val + g_mult / self._penalty >= 0. else 0.0
            for g, g_mult, g_val, g_diff in zip(self._gs, self._g_mults, g_vals, g_diff_vals))
        h_hess_sum = sum(
            2.0 * torch.outer(h_diff, h_diff) + 2.0 * (h_val / self._penalty) * h.hess(p, cfg, *args)
            for h, h_mult, h_val, h_diff in zip(self._hs, self._h_mults, h_vals, h_diff_vals)
        )
        aug_hess_value = self._f.hess(p, cfg, *args) + self._penalty / 2.0 * (g_hess_sum + h_hess_sum)
        return aug_hess_value

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
    def penalty(self, penalty: float):
        self._penalty = penalty

    @g_mults.setter
    def g_mults(self, g_mults: List[float]):
        self._g_mults = g_mults

    @h_mults.setter
    def h_mults(self, h_mults: List[float]):
        self._h_mults = h_mults


def ralm(
        f: MfldFunc,
        gs: List[MfldFunc],
        hs: List[MfldFunc],
        p0: torch.Tensor,
        mfld: ComputeMfld,
        cfg: RalmCfg,
        *args: *FuncArgs
) -> RalmResult:
    g_mults = cfg.g_mult_0 * torch.ones((len(gs),))
    h_mults = cfg.h_mult_0 * torch.ones((len(hs),))

    p = p0.clone()
    sigma: Optional[torch.Tensor] = None  # no value at start

    # track the various histories over time
    p_hist = []
    f_hist = []
    gs_hist = []
    hs_hist = []
    acc_hist = []
    penalty_hist = []
    g_mults_hist = []
    h_mults_hist = []
    subsolver_iters_hist = []

    penalty = cfg.penalty_0
    acc_tol = cfg.acc_tol_0

    subsolver_cfg = copy(cfg.subsolver_cfg)  # we will update this over time

    # sets up the augmented lagrangian function which will be minimized by the selected subsolver optimization method
    # prior to the update of the other parameters and the lagrangian multipliers
    augm_lagr = AugmentedLagrangian(f, gs, hs, g_mults.tolist(), h_mults.tolist(), penalty)

    successfully_converged = False
    idx_counter = 0

    for idx in range(cfg.max_iters):
        # updates the expected accuracy inside the subsolver
        subsolver_cfg.criterion_eps = acc_tol  # updates the expected accuracy inside the subsolver

        # updates the subproblem
        augm_lagr.g_mults = g_mults.tolist()
        augm_lagr.h_mults = h_mults.tolist()
        augm_lagr.penalty = penalty

        # attempts to solve the subproblem
        subsolver_result = cfg.subsolver_method(augm_lagr, p, mfld, subsolver_cfg, *args)
        if not subsolver_result.success:
            print(f"subsolver failed: {subsolver_result}")
            return RalmResult(
                success=False,
                p=subsolver_result.p,
                iters=idx + 1,
                history=RalmHistory(
                    p_hist=torch.stack(p_hist),
                    f_hist=torch.tensor(f_hist),
                    gs_hist=torch.stack(gs_hist),
                    hs_hist=torch.stack(hs_hist),
                    acc_hist=torch.tensor(acc_hist),
                    penalty_hist=torch.tensor(penalty_hist),
                    g_mults_hist=torch.stack(g_mults_hist),
                    h_mults_hist=torch.stack(h_mults_hist),
                    subsolver_iters_hist=subsolver_iters_hist,
                )
            )

        # assume that the subsolver was successful after this point
        next_p = subsolver_result.p

        # print(f"next_p: {next_p}")

        # check convergence criterion
        if mfld.dist(p, next_p) < cfg.min_step and acc_tol <= cfg.acc_tol_min:
            successfully_converged = True

        # convergence has not been achieved, proceed with updates

        # if convergence has not been achieved then we update the multipliers, etc.
        current_h_vals = torch.tensor([h.value(p, mfld, *args) for h in hs])

        next_g_vals = torch.tensor([g.value(next_p, mfld, *args) for g in gs])
        next_h_vals = torch.tensor([h.value(next_p, mfld, *args) for h in hs])

        next_h_mults = torch.clamp(h_mults + penalty * torch.tensor(next_h_vals), cfg.h_mult_min, cfg.h_mult_max)
        next_g_mults = torch.clamp(g_mults + penalty * torch.tensor(next_g_vals), cfg.g_mult_min, cfg.g_mult_max)

        # print(f"\tnext_g_mults: {next_g_mults}")
        # print(f"\tnext_h_mults: {next_h_mults}")

        next_sigma = torch.maximum(next_g_vals, -g_mults / penalty)
        next_acc = max(cfg.acc_tol_min, cfg.acc_decay * acc_tol)

        num_gs = len(gs)  # for clarity
        num_hs = len(hs)

        if idx == 0 or (
                # NOTE: relies on short-circuiting for sigma to not be None at this point (only present here for typing)
                sigma is not None and
                torch.any(
                    torch.maximum(
                        torch.unsqueeze(torch.tensor(next_h_vals).abs(), -1).repeat((1, num_gs)),  # stacks horiz
                        torch.unsqueeze(next_sigma.abs(), 0).repeat((num_hs, 1))  # stacks vertically
                    ) <= cfg.ratio * torch.maximum(
                        torch.unsqueeze(torch.tensor(current_h_vals).abs(), -1).repeat((1, num_gs)),  # stacks horiz
                        torch.unsqueeze(sigma.abs(), 0).repeat((num_hs, 1))  # stacks vertically
                    )
                )
        ):
            next_penalty = penalty  # no change in penalty
        else:
            next_penalty = cfg.penalty_growth * penalty

        # update the values
        p = next_p

        g_mults = next_g_mults
        h_mults = next_h_mults

        sigma = next_sigma
        acc_tol = next_acc
        penalty = next_penalty

        # update the histories

        next_f_val = f.value(next_p, mfld, *args)

        p_hist.append(p)
        f_hist.append(next_f_val)
        gs_hist.append(next_g_vals)
        hs_hist.append(next_h_vals)
        acc_hist.append(next_acc)
        penalty_hist.append(penalty)
        g_mults_hist.append(g_mults)
        h_mults_hist.append(h_mults)
        subsolver_iters_hist.append(subsolver_result.iters)

        # breaks after updating history
        idx_counter += 1
        if successfully_converged:
            break

    return RalmResult(
        success=successfully_converged,
        p=p,
        iters=idx_counter,
        history=RalmHistory(
            p_hist=torch.stack(p_hist),
            f_hist=torch.tensor(f_hist),
            gs_hist=torch.stack(gs_hist),
            hs_hist=torch.stack(hs_hist),
            acc_hist=torch.tensor(acc_hist),
            penalty_hist=torch.tensor(penalty_hist),
            g_mults_hist=torch.stack(g_mults_hist),
            h_mults_hist=torch.stack(h_mults_hist),
            subsolver_iters_hist=subsolver_iters_hist,
        )
    )
