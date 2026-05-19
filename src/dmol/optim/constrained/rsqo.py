from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import torch
import numpy as np
from torch import hub

from diff_mfld.geometry.funcs import MfldFunc, FuncArgs
from diff_mfld.mfld import ComputeMfld
from optim.results import ConstrSolverCfg


def _subproblem_grad(delta_x: np.ndarray, g_k: np.ndarray, b_k: np.ndarray, grad_f_k: np.ndarray):
    return b_k.T @ g_k @ delta_x + g_k.T @ grad_f_k


def _solve_subproblem(x_k: torch.Tensor, g_k: torch.Tensor, b_k: torch.Tensor, gs_k: List[float], hs_k: List[float],
                      grad_f_k: torch.Tensor, grad_gs_k: List[torch.Tensor], grad_hs_k: List[torch.Tensor],
                      max_iters: int, abs_acc: float, damp: float) -> Tuple[torch.Tensor, List[float], List[float]]:
    # convert to numpy (types have to be manually specified for some reason)
    g_k: np.ndarray = g_k.detach().numpy()
    b_k: np.ndarray = b_k.detach().numpy()
    grad_f_k: np.ndarray = grad_f_k.detach().numpy()
    grad_gs_k: List[np.ndarray] = [grad_g_k.detach().numpy() for grad_g_k in grad_gs_k]
    grad_hs_k: List[np.ndarray] = [grad_h_k.detach().numpy() for grad_h_k in grad_hs_k]

    delta_x = np.zeros_like(x_k)
    for idx in range(max_iters):
        # implements proximal update shceme through projected gradient descent
        prob_grad = _subproblem_grad(delta_x, g_k, b_k, grad_f_k)

        forw_step = delta_x - damp * prob_grad

        # performs all the specified projections
        # NOTE: the metric in this case is folded in so we can just treat it as linear problem
        proj_step = forw_step
        for g, grad_g in zip(gs_k, grad_gs_k):
            proj_step = _project_halfspace(grad_g @ g_k, proj_step, -g)
        for h, grad_h in zip(hs_k, grad_hs_k):
            proj_step = _project_hyperplane(grad_h @ g_k, proj_step, -h)
        delta_x_upd = proj_step  # renamed for clarity

        # exits early if the accuracy of any component is below the required threshold
        abs_err = np.abs(delta_x_upd - delta_x)
        if np.max(abs_err) <= abs_acc:
            delta_x = delta_x_upd
            break

        delta_x = delta_x_upd

    # finds the active inequality constraints
    active_ineqs = [i for i, (gi, grad_gi) in enumerate(zip(gs_k, grad_gs_k)) if gi + grad_gi @ g_k @ delta_x < 0]

    sub_f_grad = _subproblem_grad(delta_x, g_k, b_k, grad_f_k)
    active_sub_gs_grads = [grad_gs_k[i] @ g_k for i in active_ineqs]
    sub_hs_grads = [grad_hi @ g_k for grad_hi in grad_hs_k]

    if len(active_sub_gs_grads) == 0 and len(sub_hs_grads) == 0:
        # no multipliers
        g_mults = np.zeros((len(gs_k)))
        h_mults = np.zeros((0,))  # shape as hs always have gradients
    else:
        # need to solve for the multipliers
        constr_grads = np.column_stack((*active_sub_gs_grads, *sub_hs_grads))
        active_mults = np.linalg.pinv(constr_grads) @ -sub_f_grad

        # separates out all the multipliers
        g_mults = np.zeros((len(gs_k)))
        for i, active_idx in enumerate(active_ineqs):
            g_mults[active_idx] = active_mults[i]

        h_mults = active_mults[len(gs_k):]

    # assuming we're at (or very close to) an optimal solution so we'll just clamp the g_mults to be positive
    g_mults = np.clip(g_mults, 0., np.inf)
    return torch.tensor(delta_x), g_mults.tolist(), h_mults.tolist()


def _project_halfspace(c: np.ndarray, x: np.ndarray, b: float) -> np.ndarray:
    # projection onto halfspace courtesy of course notes
    if c @ x <= b:
        return x  # already in halfspace
    else:
        x_proj = x + (b - c @ x) / (c @ c) * c
        return x_proj


def _project_hyperplane(c: np.ndarray, x: np.ndarray, b: float) -> np.ndarray:
    x_proj = x + (b - c @ x) / (c @ c) * c
    return x_proj


class RiemMerit(MfldFunc):
    # defined as an L1 Riemannian penalty method

    def __init__(self, f: MfldFunc, gs: List[MfldFunc], hs: List[MfldFunc], penalty: float):
        self._f = f
        self._gs = gs
        self._hs = hs
        self._penalty = penalty

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, penalty: float):
        self._penalty = penalty

    def value(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        result = self._f.value(p, cfg, *args)
        for g in self._gs:
            result += self._penalty * torch.maximum(torch.tensor(0.0), g.value(p, cfg, *args))
        for h in self._hs:
            result += self._penalty * torch.abs(h.value(p, cfg, *args))
        return result

    def diff(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        raise NotImplementedError()  # only need the value

    def hess(self, p: torch.Tensor, cfg: ComputeMfld, *args: *FuncArgs) -> torch.Tensor:
        raise NotImplementedError()  # only need the value


def _backtracking(x_k: torch.Tensor, delta_x_optimal: torch.Tensor, b_k: torch.Tensor, g_k: torch.Tensor, beta: float,
                  gamma: float, merit_method: MfldFunc, mfld: ComputeMfld, *args: *FuncArgs) -> float:
    r = 0

    oper_inner_prod = gamma * (b_k @ delta_x_optimal) @ g_k @ delta_x_optimal
    orig_penalty = merit_method.value(x_k, mfld, args)

    while True:
        beta_r = beta ** r
        upd_x_k = mfld.exp(x_k, beta_r * delta_x_optimal)

        upd_penalty = merit_method.value(upd_x_k, mfld, *args)

        lhs = beta_r * oper_inner_prod
        rhs = orig_penalty - upd_penalty

        if lhs <= rhs:
            return beta_r
        else:
            r += 1


@dataclass
class RsqoHistory:
    p_hist: torch.Tensor

    f_hist: torch.Tensor
    gs_hist: torch.Tensor
    hs_hist: torch.Tensor

    g_mults_hist: torch.Tensor
    h_mults_hist: torch.Tensor

    penalty_hist: torch.Tensor


@dataclass
class RsqoResult:
    success: bool
    p: torch.Tensor
    iters: int
    history: RsqoHistory


@dataclass
class RsqoCfg(ConstrSolverCfg):
    beta: float = 0.4  # in (0, 1)
    gamma: float = 0.90  # in (0, 1)
    penalty_0: float = 1.2  # > 0
    eps: float = 0.1  # > 0
    symm_lin_oper: Callable[[torch.Tensor], torch.Tensor] = field(default=lambda p: 1.*torch.eye(p.shape[0]))
    max_iters: int = 1000

    subproblem_max_iters = 100
    subproblem_abs_acc: float = 0.01  # 3 percent
    subproblem_damp: float = 0.4

    min_step = 0.01


def rsqo(
        f: MfldFunc,
        gs: List[MfldFunc],
        hs: List[MfldFunc],
        p0: torch.Tensor,
        mfld: ComputeMfld,
        cfg: RsqoCfg,
        *args: *FuncArgs
) -> RsqoResult:
    x = p0  # to stick with notation used in the paper
    penalty = cfg.penalty_0
    merit = RiemMerit(f=f, gs=gs, hs=hs, penalty=penalty)

    # track the various histories over time
    p_hist = []
    f_hist = []
    gs_hist = []
    hs_hist = []
    g_mults_hist = []
    h_mults_hist = []
    penalty_hist = []

    successfully_converged = False
    idx_counter = 0

    for idx in range(cfg.max_iters):
        metric = mfld.mfld.metric(x)

        g_k = metric.mat
        b_k = cfg.symm_lin_oper(x)

        gs_k = [g.value(x, mfld, *args).item() for g in gs]
        hs_k = [h.value(x, mfld, *args).item() for h in hs]

        grad_f_k = metric.sharp(f.diff(x, mfld, *args))
        grad_gs_k = [metric.sharp(g.diff(x, mfld, *args)) for g in gs]
        grad_hs_k = [metric.sharp(h.diff(x, mfld, *args)) for h in hs]

        # solves the subproblem
        delta_x_optimal, g_mults, h_mults = _solve_subproblem(x, g_k, b_k,
                                                              gs_k, hs_k, grad_f_k, grad_gs_k, grad_hs_k,
                                                              cfg.subproblem_max_iters, cfg.subproblem_abs_acc,
                                                              cfg.subproblem_damp)

        # computes the penalty (note that this handles all the cases for provided constraints)
        combined = []
        combined.extend(g_mults)
        combined.extend(abs(h) for h in h_mults)
        vk = max(combined)
        if vk > penalty:
            penalty = vk + cfg.eps

        # gets step length from backtracking
        merit.penalty = penalty
        alpha_k = _backtracking(x, delta_x_optimal, b_k, g_k, cfg.beta, cfg.gamma, merit, mfld, *args)

        # estimate the updated position using the "retraction" which in this case is the exponential map
        upd_x = mfld.exp(x, alpha_k * delta_x_optimal)

        if mfld.dist(x, upd_x) < cfg.min_step:
            successfully_converged = True

        # update the histories

        next_f_val = f.value(upd_x, mfld, *args)
        next_g_vals = torch.tensor([g.value(upd_x, mfld, *args) for g in gs])
        next_h_vals = torch.tensor([h.value(upd_x, mfld, *args) for h in hs])
        next_gs_mults = torch.tensor(g_mults)  # evaluated during this loop
        next_hs_mults = torch.tensor(h_mults)
        next_penalty = penalty  # in the sense that it is the value at the start of each loop

        p_hist.append(upd_x)
        f_hist.append(next_f_val)
        gs_hist.append(next_g_vals)
        hs_hist.append(next_h_vals)
        g_mults_hist.append(next_gs_mults)
        h_mults_hist.append(next_hs_mults)
        penalty_hist.append(next_penalty)

        # prepare for next iteratoin
        x = upd_x

        # breaks after updating history
        idx_counter += 1
        if successfully_converged:
            break

    return RsqoResult(
        success=True,
        p=x,
        iters=idx_counter,
        history=RsqoHistory(
            p_hist=torch.stack(p_hist),
            f_hist=torch.tensor(f_hist),
            gs_hist=torch.stack(gs_hist),
            hs_hist=torch.stack(hs_hist),
            g_mults_hist=torch.stack(g_mults_hist),
            h_mults_hist=torch.stack(h_mults_hist),
            penalty_hist=torch.tensor(penalty_hist),
        ))
