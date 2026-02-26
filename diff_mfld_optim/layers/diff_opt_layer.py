import torch

from torch.func import jacrev
from torch.autograd.function import Function
from torch.nn import Module

from multiprocessing.dummy import Pool  # threads

from typing import List, Tuple, Any, Optional

from diff_mfld_optim.geometry.funcs import FuncArgs, MfldFunc


from diff_mfld_optim.optim.constrained import (
    ConstrainedSolverCfg,
    ConstrainedSolverMethod,
    ConstrainedSolverResult,
)

from diff_mfld_optim.mfld_util import MfldCfg


class DiffMfldOptimProblem(Function):
    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,  # starting point on manifold
        f: MfldFunc,  # cost function
        gs: List[MfldFunc],  # inequality constraint functions
        hs: List[MfldFunc],  # equality constraint functions
        mfld_cfg: MfldCfg,
        solve_cfg: ConstrainedSolverCfg,
        method: ConstrainedSolverMethod,
        pre_compute_result: Optional[ConstrainedSolverResult],
        *func_args: *FuncArgs,  # additional args provided to the f, g, h
    ) -> torch.Tensor:
        # if a solver result is provided (in the case that this is applied
        # after running over the batch with multiprocessing) then we just use
        # the result directly, if not, then we solve the optimization problem

        result: ConstrainedSolverResult = (
            pre_compute_result
            if pre_compute_result is not None
            else method(f, gs, hs, p0, mfld_cfg, solve_cfg, *func_args)
        )
        if not result.success:
            raise ValueError(
                "Differentiable manifold optimization layer failed to "
                f"converge to a solution: {result}"
            )
        p_optimal = result.p

        # print(f"p_optimal={p_optimal}")

        # offloads computing the jacobian of the solution map for use in
        # backpropagation to the backwards pass given the high computational
        # load (as we don't want to compute it unnecessarily when just only
        # using this layer for inferencing)

        ctx.save_for_backward(p0, p_optimal)

        ctx.f = f
        ctx.gs = gs
        ctx.g_mults = result.g_mults
        ctx.g_vals = result.gs_eval
        ctx.mfld_cfg = mfld_cfg
        ctx.func_args = func_args

        # final result is the optimized point
        return p_optimal

    @staticmethod
    def backward(ctx, grad_output):
        (p, p_optimal) = ctx.saved_tensors

        f: MfldFunc
        gs: List[MfldFunc]
        g_mults: torch.Tensor
        g_vals: torch.Tensor
        mfld_cfg: MfldCfg

        (f, gs, g_mults, g_vals, mfld_cfg, func_args) = (
            ctx.f,  # cost function
            ctx.gs,  # inequality functions
            ctx.g_mults,  # lagrangian multipliers of inequality constraints
            ctx.g_vals,  # value of inequality constraints
            ctx.mfld_cfg,  # manifold configuration (connection, metric, etc.)
            ctx.func_args,  # additional args for cost and constraints
        )

        n = len(p)  # dimension of the underlying manifold

        # evaluates the original solution map dual (tensor accepting tangent
        # vector of the optimization curve twice) at the optimized point
        hessian_f: torch.Tensor = f.hess(p_optimal, mfld_cfg, *func_args)
        partial_gs: List[torch.Tensor] = [
            g.diff(p_optimal, mfld_cfg, *func_args) for g in gs
        ]
        hessian_gs: List[torch.Tensor] = [
            g.hess(p_optimal, mfld_cfg, *func_args) for g in gs
        ]

        soln_map_dual: torch.Tensor = hessian_f
        for g_mult, hessian_g in zip(g_mults, hessian_gs):
            soln_map_dual += g_mult * hessian_g
        for g_mult, g_val, partial_g in zip(g_mults, g_vals, partial_gs):
            soln_map_dual += g_mult / g_val * torch.outer(partial_g, partial_g)

        # parallel transport component to relate the second argument to the
        # original tangent space of the non-optimal point
        conn_coeffs_p: torch.Tensor = mfld_cfg.conn(p)
        v_at_p: torch.Tensor = mfld_cfg.log_method(p, p_optimal, mfld_cfg.conn)

        parll_tranp = torch.eye(n) - torch.tensordot(conn_coeffs_p, v_at_p, ([1], [0]))

        # finds the total solution map jacobian which accepts a tangent vector
        # at the non-optimal point and produces a parallel tangent vector at
        # the optimal point of the problem
        g_inv_p_optimal: torch.Tensor = mfld_cfg.metric_field(p_optimal).inv.mat
        soln_map_jacob: torch.Tensor = -torch.tensordot(
            torch.tensordot(g_inv_p_optimal, soln_map_dual, ([1], [0])),
            parll_tranp,
            ([1], [0]),
        )

        return soln_map_jacob * grad_output, *[None for _ in range(7 + len(func_args))]


class DiffMfldOptimLayer(Module):
    def __init__(
        self,
        f: MfldFunc,
        gs: List[MfldFunc],
        hs: List[MfldFunc],
        mfld_cfg: MfldCfg,
        solve_cfg: ConstrainedSolverCfg,
        method: ConstrainedSolverMethod,
    ):
        super().__init__()
        self.f = f
        self.gs = gs
        self.hs = hs
        self.mfld_cfg = mfld_cfg
        self.solve_cfg = solve_cfg
        self.method = method

    def forward(
        self,
        p0: torch.Tensor,  # initial point on manifold
        batched_args: Tuple[torch.Tensor],  # arguments to apply batching rules to
        args: tuple[Any],  # arguments passed as-in to optim problem
        pool: Optional[Pool],  # multiprcoessing pool to speed up batching
    ):
        is_batched = len(p0.shape) > 1
        if not is_batched:
            p_optimal = DiffMfldOptimProblem.apply(
                p0,
                self.f,
                self.gs,
                self.hs,
                self.mfld_cfg,
                self.solve_cfg,
                self.method,
                *batched_args,
                *args,
            )
            return p_optimal

        # print(f"Starting pool run!")

        # have to handle the multiprocessing case where the batched arguments
        # are indexed along their first dimension and fed to separate instances
        # of the optimization problem
        num_batches = p0.shape[0]
        p_optimal_batched = torch.zeros_like(p0)

        if pool is None:
            for i in range(num_batches):
                p_optimal_batched[i, :] = DiffMfldOptimProblem.apply(
                    p0[i, :],
                    self.f,
                    self.gs,
                    self.hs,
                    self.mfld_cfg,
                    self.solve_cfg,
                    self.method,
                    None,  # executes the optimization process
                    *[arg[i, :] for arg in batched_args],
                    *args,
                )
        else:
            # solves the optimization problem for all the batches included in
            # the inputs using the multiprocessing pool for maximum speed (but
            # we can't directly apply the custom torch function this way as the
            # autograd compute graph construction cannot cross processing
            # bounds so we run the computation separately)

            results: List[ConstrainedSolverResult] = list(
                pool.map(
                    lambda args: self.method(*args),
                    [
                        (
                            self.f,
                            self.gs,
                            self.hs,
                            p0[i, :].detach(),
                            self.mfld_cfg,
                            self.solve_cfg,
                            *[arg[i, :] for arg in batched_args],
                            *args,
                        )
                        for i in range(num_batches)
                    ],
                )
            )

            # we still have to use the optimization function to ensure the
            # backwards pass is correctly setup but we have moved the expensive
            # optimization problem to a multiprocessing data setup
            for i in range(num_batches):
                p_optimal_batched[i, :] = DiffMfldOptimProblem.apply(
                    p0[i, :],  # must provide this from args to connect it via backprop
                    self.f,
                    self.gs,
                    self.hs,
                    self.mfld_cfg,
                    self.solve_cfg,
                    self.method,
                    results[i],  # uses the pre-computed optimization results
                    *[arg[i, :] for arg in batched_args],
                    *args,
                )
        return p_optimal_batched
