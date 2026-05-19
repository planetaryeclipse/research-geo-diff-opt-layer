import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

import torch

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.results import SubsolverCfg, SubsolverCriterion, ConstrSolverCfg
from optim.subsolver_methods import SubsolverMethod
from optim.subsolvers.rgd import RiemGradDescentCfg


# implements the Riemannian Exact Penalty Method via Smoothing (REPMS) described in "Simple Algorithms for Optimization
# on Riemannian Manifolds with Constraints"

class PseudoHuberLoss(MfldFunc):
    def __init__(self, f: MfldFunc, u: float):
        self._f = f
        self._u = u

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        f_val = self._f.value(p, cfg, *args)
        if f_val <= 0.0:
            return torch.tensor(0.0)
        elif f_val <= self._u:
            return f_val ** 2 / (2.0 * self._u)
        else:
            return f_val - self._u / 2.0

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        f_val = self._f.value(p, cfg, *args)
        if f_val <= 0.0:
            return torch.tensor(0.0)
        elif f_val <= self._u:
            return f_val.item() / self._u * self._f.diff(p, cfg, *args)
        else:
            return self._f.diff(p, cfg, *args)

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        f_val = self._f.value(p, cfg, *args)
        if f_val <= 0.0:
            return torch.tensor(0.0)
        elif f_val <= self._u:
            f_diff = self._f.diff(p, cfg, *args)
            return 1.0 / self._u * torch.outer(f_diff, f_diff) + 1.0 / self._u * self._f.hess(p, cfg, *args)
        else:
            return self._f.hess(p, cfg, *args)

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u):
        self._u = u


class SubproblemLSE(MfldFunc):
    def __init__(self, f: MfldFunc, gs: List[MfldFunc], hs: List[MfldFunc], u: float, penalty: float):
        self._f = f
        self._gs = gs
        self._hs = hs
        self._u = u
        self._penalty = penalty

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.value(p, cfg, *args)
        for g in self._gs:
            result += self._penalty * self._u * torch.log(1.0 + torch.exp(g.value(p, cfg, *args) / self._u))
        for h in self._hs:
            result += self._penalty * self._u * torch.log(
                torch.exp(h.value(p, cfg, *args) / self._u) + torch.exp(-h.value(p, cfg, *args) / self._u))
        return result

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.diff(p, cfg, *args)
        for g in self._gs:
            exp_g = torch.exp(g.value(p, cfg, *args) / self._u)

            # print(g.value(p, cfg, *args) / self._u)
            # print(g.value(p, cfg, *args))
            # print(f"u: {self._u}")
            # print(f"exp_g: {exp_g}")

            result += self._penalty * exp_g / (1.0 + exp_g) * g.diff(p, cfg, *args)
        for h in self._hs:
            exp_pos_h = torch.exp(h.value(p, cfg, *args) / self._u)
            exp_neg_h = torch.exp(h.value(p, cfg, *args) / self._u)
            diff_h = h.diff(p, cfg, *args)

            result += self._penalty * (exp_pos_h + exp_neg_h) * exp_pos_h * diff_h
            result += -self._penalty * (exp_pos_h + exp_neg_h) * exp_neg_h * diff_h

        # print(f"lse diff: {result}")

        return result

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.hess(p, cfg, *args)
        for g in self._gs:
            exp_g = torch.exp(g.value(p, cfg, *args) / self._u)
            diff_g = g.diff(p, cfg, *args)

            result += self._penalty * exp_g / self._u / (1 + exp_g) * torch.outer(diff_g, diff_g)
            result += -self._penalty * exp_g / (1 + exp_g) ** 2 * exp_g / self._u * torch.outer(diff_g, diff_g)
            result += self._penalty * exp_g / (1 + exp_g) * g.hess(p, cfg, *args)
        for h in self._hs:
            exp_pos_h = torch.exp(h.value(p, cfg, *args) / self._u)
            exp_neg_h = torch.exp(h.value(p, cfg, *args) / self._u)
            diff_h = h.diff(p, cfg, *args)
            hess_h = h.hess(p, cfg, *args)

            result += self._penalty * exp_pos_h / self._u * exp_pos_h * torch.outer(diff_h, diff_h)
            result += -self._penalty * exp_neg_h / self._u * exp_pos_h * torch.outer(diff_h, diff_h)
            result += (exp_pos_h + exp_neg_h) * exp_pos_h / self._u * torch.outer(diff_h, diff_h)
            result += (exp_pos_h + exp_neg_h) * exp_pos_h * hess_h

            result += -self._penalty * exp_pos_h / self._u * exp_neg_h * torch.outer(diff_h, diff_h)
            result += +self._penalty * exp_neg_h / self._u * exp_neg_h * torch.outer(diff_h, diff_h)
            result += -(exp_pos_h + exp_neg_h) * exp_neg_h / self._u * torch.outer(diff_h, diff_h)
            result += -(exp_pos_h + exp_neg_h) * exp_neg_h * hess_h
        return result


class SubproblemLQH(MfldFunc):
    def __init__(self, f: MfldFunc, gs: List[MfldFunc], hs: List[MfldFunc], u: float, penalty: float):
        self._f = f
        self._gs = gs
        self._hs = hs
        self._u = u
        self._penalty = penalty

        self._gs_pseudo_losses = [PseudoHuberLoss(g, u) for g in gs]

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.value(p, cfg, *args)
        for g_loss in self._gs_pseudo_losses:
            g_loss.u = self._u
            result += self._penalty * g_loss.value(p, cfg, *args)
        for h in self._hs:
            result += self._penalty * torch.sqrt(h.value(p, cfg, *args) ** 2 + self._u ** 2)
        return result

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.diff(p, cfg, *args)
        for g_loss in self._gs_pseudo_losses:
            g_loss.u = self._u
            result += self._penalty * g_loss.diff(p, cfg, *args)
        for h in self._hs:
            h_val = h.value(p, cfg, *args)
            result += self._penalty / torch.sqrt(h_val ** 2 + self._u ** 2) * h_val * h.diff(p, cfg, *args)
        return result

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.hess(p, cfg, *args)
        for g_loss in self._gs_pseudo_losses:
            g_loss.u = self._u
            result += self._penalty * g_loss.hess(p, cfg, *args)
        for h in self._hs:
            h_val = h.value(p, cfg, *args)
            h_diff = h.diff(p, cfg, *args)

            result += -self._penalty / torch.float_power(h_val ** 2 + self._u ** 2, 1.5 / 2) * h_val ** 2 * torch.outer(
                h_diff, h_diff)
            result += self._penalty / (h_val ** 2 + self._u ** 2) * torch.outer(h_diff, h_diff)
            result += self._penalty / (h_val ** 2 + self._u ** 2) * h.hess(p, cfg, *args)
        return result

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u: float):
        self._u = u

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: float):
        self._penalty = penalty


class RepmMode(Enum):
    LSE = auto()
    LQH = auto()


@dataclass
class RepmHistory:
    p_hist: torch.Tensor

    f_hist: torch.Tensor
    gs_hist: torch.Tensor
    hs_hist: torch.Tensor

    acc_hist: torch.Tensor
    penalty_hist: torch.Tensor
    approx_acc_hist: torch.Tensor

    subsolver_iters_hist: List[int]


@dataclass
class RepmCfg(ConstrSolverCfg):
    acc_tol_min: float = 1e-3  # epsilon
    acc_tol_0: float = 1e-2
    acc_decay: float = 0.9  # (0, 1)

    penalty_0: float = 0.5  # rho
    penalty_growth: float = 1.1  # > 1
    penalty_update: bool = True  # whether to update the penalty
    penalty_constr_threshold: float = 1e-6  # update if contraint violates this threshold

    approx_acc_min: float = 1e-3
    approx_acc_0: float = 1e-2  # u
    approx_acc_decay: float = 0.9  # (0, 1)

    min_step: float = 0.02

    mode: RepmMode = RepmMode.LSE

    subsolver_method: SubsolverMethod = SubsolverMethod.RIEM_GRAD_DESCENT
    subsolver_cfg: SubsolverCfg = field(
        default_factory=RiemGradDescentCfg,
    )  # must be type corresponding to the subsolver method
    max_iters: int = 1000


@dataclass
class RepmResult:
    success: bool
    p: torch.Tensor
    iters: int
    history: RepmHistory


def repm(f: MfldFunc,
         gs: List[MfldFunc],
         hs: List[MfldFunc],
         p0: torch.Tensor,
         mfld: ComputeMfld,
         cfg: RepmCfg,
         *args: *FuncArgs) -> RepmResult:
    p = p0.clone()
    subsolver_cfg = copy.copy(cfg.subsolver_cfg)  # we will update this over time

    # track the variou shistories over time
    p_hist = []
    f_hist = []
    gs_hist = []
    hs_hist = []
    acc_hist = []
    penalty_hist = []
    approx_acc_hist = []
    subsolver_iters_hist = []

    acc_tol = cfg.acc_tol_0
    penalty = cfg.penalty_0
    approx_acc = cfg.approx_acc_0  # u

    # sets up the correct subproblem
    q_subproblem = None
    if cfg.mode == RepmMode.LSE:
        q_subproblem = SubproblemLSE(f, gs, hs, approx_acc, penalty)
    elif cfg.mode == RepmMode.LQH:
        q_subproblem = SubproblemLQH(f, gs, hs, approx_acc, penalty)

    successfully_converged = False
    idx_counter = 0

    for idx in range(cfg.max_iters):
        # updates the expected accuracy inside the subsolver (and ensures uses norm stopping criterion)
        subsolver_cfg.criterion_eps = acc_tol
        subsolver_cfg.criterion = SubsolverCriterion.NORM  # required by REPM

        # updates the subproblem
        q_subproblem.u = approx_acc
        q_subproblem.penalty = penalty

        # attempts to solve the subproblem
        subsolver_result = cfg.subsolver_method(q_subproblem, p, mfld, subsolver_cfg, *args)
        if not subsolver_result.success:
            return RepmResult(
                success=False,
                p=subsolver_result.p,
                iters=idx + 1,
                history=RepmHistory(
                    p_hist=torch.stack(p_hist) if len(p_hist) > 0 else torch.zeros((0, len(p))),
                    f_hist=torch.tensor(f_hist),
                    gs_hist=torch.stack(gs_hist) if len(gs_hist) > 0 else torch.zeros((0, len(gs))),
                    hs_hist=torch.stack(hs_hist) if len(hs_hist) > 0 else torch.zeros((0, len(hs))),
                    acc_hist=torch.tensor(acc_hist),
                    penalty_hist=torch.tensor(penalty_hist),
                    approx_acc_hist=torch.tensor(approx_acc_hist),
                    subsolver_iters_hist=subsolver_iters_hist,
                )
            )

        # assume that the subsolver was succssful after this point
        next_p = subsolver_result.p

        # print(f"next_p: {next_p}")

        # check convergence criterion
        if mfld.dist(p, next_p) < cfg.min_step and acc_tol <= cfg.acc_tol_min and approx_acc <= cfg.approx_acc_min:
            successfully_converged = True

        # convergence has not beeen achieved, proceed with updates

        next_acc_tol = max(cfg.acc_tol_min, cfg.acc_decay * acc_tol)
        next_approx_acc = max(cfg.approx_acc_min, cfg.approx_acc_decay * approx_acc)

        next_g_vals = torch.tensor([g.value(next_p, mfld, *args) for g in gs])
        next_h_vals = torch.tensor([h.value(next_p, mfld, *args) for h in hs])

        if cfg.penalty_update and (
                idx == 0 or torch.max(next_g_vals.abs()) >= cfg.penalty_constr_threshold or torch.max(
            next_h_vals.abs()) >= cfg.penalty_growth):
            next_penalty = cfg.penalty_growth * penalty
        else:
            next_penalty = penalty  # no change in penalty

        # update the values
        p = next_p
        acc_tol = next_acc_tol
        penalty = next_penalty
        approx_acc = next_approx_acc

        # update the histories
        next_f_val = f.value(next_p, mfld, *args)

        p_hist.append(next_p)
        f_hist.append(next_f_val)
        gs_hist.append(next_g_vals)
        hs_hist.append(next_h_vals)
        acc_hist.append(next_acc_tol)
        penalty_hist.append(next_penalty)
        approx_acc_hist.append(next_approx_acc)
        subsolver_iters_hist.append(subsolver_result.iters)

        # breaks after updating history
        idx_counter += 1
        if successfully_converged:
            break

    return RepmResult(
        success=successfully_converged,
        p=p,
        iters=idx_counter,
        history=RepmHistory(
            p_hist=torch.stack(p_hist),
            f_hist=torch.tensor(f_hist),
            gs_hist=torch.stack(gs_hist),
            hs_hist=torch.stack(hs_hist),
            acc_hist=torch.tensor(acc_hist),
            penalty_hist=torch.tensor(penalty_hist),
            approx_acc_hist=torch.tensor(approx_acc_hist),
            subsolver_iters_hist=subsolver_iters_hist,
        )
    )
