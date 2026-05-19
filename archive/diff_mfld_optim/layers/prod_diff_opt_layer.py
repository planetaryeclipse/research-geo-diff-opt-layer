import torch

from torch.func import jacrev
from torch.autograd.function import Function
from torch.nn import Module

from multiprocessing.pool import Pool
from typing import List, Tuple, Any, Optional

from diff_mfld_optim.mfld_util import MfldCfg
from diff_mfld_optim.optim.subsolver import OptimFunc, FuncArgs
from diff_mfld_optim.optim.constrained import (
    ConstrainedSolverCfg,
    ConstrainedSolverMethod,
    ConstrainedSolverResult,
)
from diff_mfld_optim.geometry.connection import Connection

from typing import Callable, List

ProdOptimFunc = Callable[[torch.Tensor, torch.Tensor, MfldCfg, *FuncArgs], torch.Tensor]


class ProdDiffMfldOptimProblem(Function):
    # this optimization layer still uses the same underlying optimizer as in
    # the non-product (as still only optimizing on one of the products) but
    # function accepts a point on the manifold of interest

    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,  # starting point on optimization manifold
        q: torch.Tensor,  # system state (on other manifold)
        _v: torch.Tensor,  # system velocity (on other manifold) -> (unused for now)
        f: ProdOptimFunc,  # cost function
        gs: List[ProdOptimFunc],  # inequality constraint functions
        hs: List[ProdOptimFunc],  # equality constraint functions
        optim_mfld_cfg: MfldCfg,  # describes optimization manifold
        prod_mfld_conn: Connection,  # describes connection on product manifold
        solve_cfg: ConstrainedSolverCfg,
        method: ConstrainedSolverMethod,
        pre_compute_result: Optional[ConstrainedSolverResult],
        *func_args: *FuncArgs,  # additional args provided for the f, g, h
    ) -> torch.Tensor:
        # performed constrained optimization (using chosen solver)

        if pre_compute_result is None:
            f_wrapped: OptimFunc = lambda p, mfld_cfg, *func_args: f(
                p, q, mfld_cfg, *func_args
            )
            gs_wrapped: List[OptimFunc] = [
                lambda p, mfld_cfg, *func_args: g(p, q, mfld_cfg, *func_args)
                for g in gs
            ]
            hs_wrapped: List[OptimFunc] = [
                lambda p, mfld_cfg, *func_args: h(p, q, mfld_cfg, *func_args)
                for h in hs
            ]
            result = method(
                f_wrapped,
                gs_wrapped,
                hs_wrapped,
                p0,
                prod_mfld_conn,
                solve_cfg,
                *func_args,
            )
        else:
            result = pre_compute_result

        if not result.success:
            raise ValueError(
                "Product differentiable manifold optimization layer failed to"
                f"converge to a solution: {result}"
            )
        p_optimal = result.p

        # we also offload the computation of the solution map jacobian (for the
        # optimization manifold) for use in backpropagation to the backwards
        # pass given the high computational load (as we don't want to compute
        # it unncessarily when just only using this layer for inferencing)

        ctx.save_for_backward(p0, q, _v, p_optimal)

        # NOTE: we are passing the product functions below (any wrapping is
        # performed in the function itself for clarity)
        ctx.f = f
        ctx.gs = gs
        ctx.g_mults = result.g_mults
        ctx.g_vals = result.gs_eval
        ctx.optim_mfld_cfg = optim_mfld_cfg
        ctx.prod_mfld_conn = prod_mfld_conn
        ctx.func_args = func_args

        # final result is the optimized point
        return p_optimal

    @staticmethod
    def backward(ctx, grad_output):
        (p, q, _v_p, p_optimal) = ctx.saved_tensors
        (f, gs, g_mults, g_vals, optim_mfld_cfg, prod_mfld_conn, func_args) = (
            ctx.f,
            ctx.gs,
            ctx.g_mults,
            ctx.g_vals,
            ctx.optim_mfld_cfg,
            ctx.prod_mfld_conn,
            ctx.func_args,
        )

        n = p.shape[0]
        m = q.shape[0]
        s = len(gs)

        # note that during optimization then the state coordinate on the other
        # manifold does not change during optimization so we reuse it

        conn_coeffs_p = prod_mfld_conn(p, q)
        conn_coeffs_p_optimal = prod_mfld_conn(p_optimal, q)

        r_p = torch.cat((p, q))
        r_p_optimal = torch.cat((p_optimal, q))

        # compute the velocity at the initial point (note that we're expecting
        # the velocity on the components corresponding to the external manifold
        # to be zero as the point remains the same)
        # NOTE: may not be identically zero due to the usage of approximate
        # logarithmic maps being employed in this framework

        # NOTE: uses the same log method as on the optimal manifold but uses
        # the connection associated with the product manifold
        w_p: torch.Tensor = optim_mfld_cfg.log_method(r_p, r_p_optimal, prod_mfld_conn)

        # we need the inverse metric on the optimizing manifold
        g_inv_p_optimal: torch.Tensor = optim_mfld_cfg.metric_field(p_optimal).inv.mat

        # need the partial derivatives with respect to both manifolds (so we
        # accept a single coordinate and partition it before feeding it in)
        f_single_input = lambda w: f(w[:n], w[n:], optim_mfld_cfg, *func_args)
        gs_single_input = [
            lambda w: g(w[:n], w[n:], optim_mfld_cfg, *func_args) for g in gs
        ]

        partial_f = jacrev(f_single_input)(r_p_optimal)
        hessian_f = jacrev(jacrev(f_single_input))(r_p_optimal)

        partial_gs = [jacrev(g)(r_p_optimal) for g in gs_single_input]
        hessian_gs = [jacrev(jacrev(g))(r_p_optimal) for g in gs_single_input]

        # only indexing on the first index as it must be compatible with the
        # metric on the optimization manifold but due to the parallel transport
        # on the product manifold then we must keep the full product manifold
        # dimension on the right indices for now

        soln_map_dual = hessian_f[:n, :] - torch.tensordot(
            partial_f, conn_coeffs_p_optimal[:, :n, :], ([0], [0])
        )
        for i in range(s):
            partial_g = partial_gs[i]
            hessian_g = hessian_gs[i]

            soln_map_dual += (
                -g_mults[i] / g_vals[i] * torch.outer(partial_g[:n], partial_g)
                + g_mults[i] * hessian_g[:n, :]
                - torch.tensordot(
                    partial_g, conn_coeffs_p_optimal[:, :n, :], ([0], [0])
                )
            )

        # as the rightmost side will be multiplied by the tangent vector on the
        # optimization manifold then we slice it but keep the first argument at
        # the full dimensionality of the product manifold (to be compatible
        # with the solution map dual)
        parallel_transp = torch.eye(n + m, n) - torch.tensordot(
            conn_coeffs_p[:, :, :n], w_p, ([1], [0])
        )

        soln_map_jacob = torch.tensordot(
            g_inv_p_optimal,
            torch.tensordot(soln_map_dual, parallel_transp, ([1], [0])),
            ([1], [0]),
        )

        return grad_output * soln_map_jacob, *[None for _ in range(10 + len(func_args))]


class ProdDiffMfldOptimLayer(Module):
    def __init__(
        self,
        f: ProdOptimFunc,
        gs: List[ProdOptimFunc],
        hs: List[ProdOptimFunc],
        optim_mfld_cfg: MfldCfg,
        prod_mfld_conn: Connection,
        solve_cfg: ConstrainedSolverCfg,
        method: ConstrainedSolverMethod,
    ):
        super().__init__()
        self.f = f
        self.gs = gs
        self.hs = hs
        self.optim_mfld_cfg = optim_mfld_cfg
        self.prod_mfld_conn = prod_mfld_conn
        self.solve_cfg = solve_cfg
        self.method = method

    def forward(
        self,
        p0: torch.Tensor,
        q: torch.Tensor,
        batched_args: Tuple[torch.Tensor],  # arguments to apply batching rules to
        args: Tuple[Any],  # arguments passed as-in to optim problem
        pool: Optional[Pool],  # multiprcoessing pool to speed up batching
        _v: torch.Tensor = None,  # velocity on the non-optimization manifold (unused)
    ):
        # NOTE: at time of writing the velocity on the external manifold _v is
        # currently unused and is therefore left undefined (but keeping as part
        # of the API for convenience)

        is_batched = len(p0.shape) > 1
        if not is_batched:
            p_optimal = ProdDiffMfldOptimProblem.apply(
                p0,
                q,
                _v,
                self.f,
                self.gs,
                self.hs,
                self.optim_mfld_cfg,
                self.prod_mfld_conn,
                self.solve_cfg,
                self.method,
                None,  # executes the optimization problem
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
                p_optimal_batched[i, :] = ProdDiffMfldOptimProblem.apply(
                    p0[i, :],
                    q[i, :],
                    _v[i, :] if _v is not None else None,
                    self.f,
                    self.gs,
                    self.hs,
                    self.optim_mfld_cfg,
                    self.prod_mfld_conn,
                    self.solve_cfg,
                    self.method,
                    None,  # executes the optimization problem
                    *[arg[i, :] for arg in batched_args],
                    *args,
                )
        else:
            # solves the optimization problem for all the batches included in
            # the inputs using the multiprocessing pool for maximum speed (but
            # we can't directly apply the custom torch function this way as the
            # autograd compute graph construction cannot cross processing
            # bounds so we run the computation for each sample in batch
            # separately)

            # unlike in the application for non-batched results, we cannot just
            # define lambda variants f, gs, and hs naively as we need to factor
            # in the fact that q is also batched so we need to recreate it for
            # each set of (p,q) values

            prod_results: List[ConstrainedSolverResult] = list(
                pool.map(
                    lambda args: self.method(*args),
                    [
                        (
                            lambda p, mfld_cfg, *func_args: self.f(
                                p, q[i, :], mfld_cfg, *func_args
                            ),
                            [
                                lambda p, mfld_cfg, *func_args: self.gs[j](
                                    p, q[i, :], mfld_cfg, *func_args
                                )
                                for j in range(len(self.gs))
                            ],
                            [
                                lambda p, mfld_cfg, *func_args: self.hs[j](
                                    p, q[i, :], mfld_cfg, *func_args
                                )
                                for j in range(len(self.hs))
                            ],
                            p0[i, :].detach(),
                            self.optim_mfld_cfg,
                            self.solve_cfg,
                            *[arg[i, :] for arg in batched_args],
                            *args,
                        )
                        for i in range(num_batches)
                    ],
                )
            )

            # combines the results into the batched output
            for i in range(num_batches):
                p_optimal_batched[i, :] = ProdDiffMfldOptimProblem.apply(
                    prod_results[i].p0,
                    q[i, :],
                    _v[i, :] if _v is not None else None,
                    self.f,
                    self.gs,
                    self.hs,
                    self.optim_mfld_cfg,
                    self.prod_mfld_conn,
                    self.solve_cfg,
                    self.method,
                    prod_results[i],  # uses the pre-computed optimization results
                    *[arg[i, :] for arg in batched_args],
                    *args,
                )
        return p_optimal_batched
