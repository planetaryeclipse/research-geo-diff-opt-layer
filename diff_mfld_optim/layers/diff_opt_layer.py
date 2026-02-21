import torch

from torch.func import jacrev
from torch.autograd.function import Function
from torch.nn import Module

from multiprocessing.pool import Pool
from typing import List, Tuple, Any, Optional

from diff_mfld_optim.optim.subsolver import OptimFunc, FuncArgs
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
        f: OptimFunc,  # cost function
        gs: List[OptimFunc],  # inequality constraint functions
        hs: List[OptimFunc],  # equality constraint functions
        mfld_cfg: MfldCfg,
        solve_cfg: ConstrainedSolverCfg,
        method: ConstrainedSolverMethod,
        *func_args: *FuncArgs,  # additional args provided the f, g, h
    ) -> torch.Tensor:
        # performs constrained optimization (using chosen solver)

        result: ConstrainedSolverResult = method(
            f, gs, hs, p0, mfld_cfg, solve_cfg, *func_args
        )
        if not result.success:
            raise ValueError(
                "Differentiable manifold optimization layer failed to "
                f"converge to a solution: {result}"
            )
        p_optimal = result.p

        # offloads computing the jacobian of the solution map for use in
        # backpropagation to the backwards pass given the high computational
        # load (as we don't want to compute it unnecessarily when just only
        # using this layer for inferencing)

        ctx.save_for_backward(p0)
        ctx.save_for_backward(p_optimal)

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
        (f, gs, g_mults, g_vals, mfld_cfg, func_args) = (
            ctx.f,  # cost function
            ctx.gs,  # inequality functions
            ctx.g_mults,  # lagrangian multipliers of inequality constraints
            ctx.g_vals,  # value of inequality constraints
            ctx.mfld_cfg,  # manifold configuration (connection, metric, etc.)
            ctx.func_args,  # additional args for cost and constraints
        )

        conn_coeffs_p = mfld_cfg.conn(p)
        v_p = mfld_cfg.log_method(p, p_optimal, mfld_cfg.conn)

        conn_coeffs_p_optimal = mfld_cfg.conn(p_optimal)
        g_inv_p_optimal = mfld_cfg.metric_field(p_optimal).inv()

        # various partial derivatives needed
        f_lambda_fn = lambda p: f(p, mfld_cfg, *func_args)
        g_lambda_fns = [lambda p: g(p, mfld_cfg, *func_args) for g in gs]

        # using jacrev composition over hessian for speed (although given this
        # is only run once per inference then we don't need the full speed that
        # is possible to be achieved through the backpropagation trick)
        partial_f = jacrev(f_lambda_fn)(p_optimal)
        hessian_f = jacrev(jacrev(f_lambda_fn))(p_optimal)

        partial_gs = [jacrev(g)(p_optimal) for g in g_lambda_fns]
        hessian_gs = [jacrev(jacrev(g))(p_optimal) for g in g_lambda_fns]

        s = len(gs)  # number of inequality constraints

        # original solution map dual (before adding parallel transport)
        soln_map_dual = hessian_f - torch.tensordot(
            partial_f, conn_coeffs_p_optimal, (0, 0)
        )
        for i in range(s):
            partial_g = partial_gs[i]
            hessian_g = hessian_gs[i]

            soln_map_dual += (
                -g_mults[i] / g_vals[i] * torch.outer(partial_g, partial_g)
                + g_mults[i] * hessian_g
                - torch.tensordot(partial_g, conn_coeffs_p_optimal, (0, 0))
            )

        # parallel transport component (note that we pulled the negative
        # back on this component for consistency with labelling)
        parallel_transp = -torch.tensordot(conn_coeffs_p, v_p, (1, 0))

        # full solution map jacobian
        soln_map_jacob = torch.tensordot(
            g_inv_p_optimal,
            torch.tensordot(soln_map_dual, parallel_transp, (1, 0)),
            (1, 0),
        )

        return grad_output * soln_map_jacob


class DiffMfldOptimLayer(Module):
    def __init__(
        self,
        f: OptimFunc,
        gs: List[OptimFunc],
        hs: List[OptimFunc],
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

        # have to handle the multiprocessing case where the batched arguments
        # are indexed along their first dimension and fed to separate instances
        # of the optimization problem
        num_batches = p0.shape[0]
        p_optimal_batched = torch.zeros_like(p0)

        if not is_batched:
            for i in range(num_batches):
                p_optimal_batched[i, :] = DiffMfldOptimProblem.apply(
                    p0[i, :],
                    self.f,
                    self.gs,
                    self.hs,
                    self.mfld_cfg,
                    self.solve_cfg,
                    self.method,
                    *[arg[i, :] for arg in batched_args],
                    *args,
                )
        else:
            # solves the optimization problem for all the batches included in
            # the inputs using the multiprocessing pool for maximum speed
            p_optimal_results = pool.imap(
                DiffMfldOptimProblem.apply,
                [
                    (
                        p0[i, :],
                        self.f,
                        self.gs,
                        self.hs,
                        self.mfld_cfg,
                        self.solve_cfg,
                        self.method,
                        *[arg[i, :] for arg in batched_args],
                        *args,
                    )
                    for i in range(num_batches)
                ],
            )

            # combines the results into the batched output
            for i in range(num_batches):
                p_optimal_batched[i, :] = p_optimal_results[i]
        return p_optimal_batched
