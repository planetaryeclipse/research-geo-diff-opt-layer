import torch

from torch.func import jacrev
from torch.autograd.function import Function
from torch.nn import Module

# from multiprocessing.pool import Pool
# from pathos.multiprocessing import ProcessingPool
from multiprocessing.dummy import Pool  # threads

from typing import List, Tuple, Any, Optional, Union

from diff_mfld_optim.optim.subsolver import OptimFunc, FuncArgs
from diff_mfld_optim.optim.constrained import (
    ConstrainedSolverCfg,
    ConstrainedSolverMethod,
    ConstrainedSolverResult,
)
from diff_mfld_optim.mfld_util import MfldCfg

from diff_mfld_optim.geometry.metric import Metric, MetricField, MetricView

import dill

import tqdm


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
        pre_compute_result: Optional[ConstrainedSolverResult],
        *func_args: *FuncArgs,  # additional args provided the f, g, h
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
        (f, gs, g_mults, g_vals, mfld_cfg, func_args) = (
            ctx.f,  # cost function
            ctx.gs,  # inequality functions
            ctx.g_mults,  # lagrangian multipliers of inequality constraints
            ctx.g_vals,  # value of inequality constraints
            ctx.mfld_cfg,  # manifold configuration (connection, metric, etc.)
            ctx.func_args,  # additional args for cost and constraints
        )

        conn_coeffs_p: torch.Tensor = mfld_cfg.conn(p)
        v_p: torch.Tensor = mfld_cfg.log_method(p, p_optimal, mfld_cfg.conn)

        conn_coeffs_p_optimal: torch.Tensor = mfld_cfg.conn(p_optimal)
        g_inv_p_optimal: torch.Tensor = mfld_cfg.metric_field(p_optimal).inv.mat

        # various partial derivatives needed
        f_lambda_fn = lambda p: f(p, mfld_cfg, *func_args)
        g_lambda_fns = [lambda p: g(p, mfld_cfg, *func_args) for g in gs]

        # using jacrev composition over hessian for speed (although given this
        # is only run once per inference then we don't need the full speed that
        # is possible to be achieved through the backpropagation trick)
        partial_f: torch.Tensor = jacrev(f_lambda_fn)(p_optimal)
        hessian_f: torch.Tensor = jacrev(jacrev(f_lambda_fn))(p_optimal)

        print(f"partial_f={partial_f}")
        print(f"hessian_f=={hessian_f}")

        partial_gs: List[torch.Tensor] = [jacrev(g)(p_optimal) for g in g_lambda_fns]
        hessian_gs: List[torch.Tensor] = [
            jacrev(jacrev(g))(p_optimal) for g in g_lambda_fns
        ]

        n = len(p)  # manifold dimension
        s = len(gs)  # number of inequality constraints

        # original solution map dual (before adding parallel transport)
        soln_map_dual: torch.Tensor = hessian_f - torch.tensordot(
            partial_f, conn_coeffs_p_optimal, ([0], [0])
        )
        

        for i in range(s):
            partial_g = partial_gs[i]
            hessian_g = hessian_gs[i]

            soln_map_dual += (
                -g_mults[i] / g_vals[i] * torch.outer(partial_g, partial_g)
                + g_mults[i] * hessian_g
                - torch.tensordot(partial_g, conn_coeffs_p_optimal, ([0], [0]))
            )

        print(f"soln_map_dual={soln_map_dual}")

        # parallel transport component (note that we pulled the negative
        # back on this component for consistency with labelling)
        parallel_transp: torch.Tensor = torch.eye(n) - torch.tensordot(
            conn_coeffs_p, v_p, ([1], [0])
        )

        print(f"parallel_transport={parallel_transp}")
        print(f"dot={torch.tensordot(soln_map_dual, parallel_transp, ([1], [0]))}")

        assert False

        # full solution map jacobian
        soln_map_jacob: torch.Tensor = torch.tensordot(
            g_inv_p_optimal,
            torch.tensordot(soln_map_dual, parallel_transp, ([1], [0])),
            ([1], [0]),
        )

        print(soln_map_jacob)

        p_grad = grad_output @ soln_map_jacob
        print(p_grad)

        return p_grad, *[None for _ in range(7 + len(func_args))]


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
                    results[i].p0,  # uses results from multiprocessed results
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
