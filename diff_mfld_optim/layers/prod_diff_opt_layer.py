import torch

from torch.func import jacrev
from torch.autograd.function import Function
from torch.nn import Module

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
        *func_args: *FuncArgs,  # additional args provided for the f, g, h
    ) -> torch.Tensor:
        # performed constrained optimization (using chosen solver)

        f_wrapped: OptimFunc = lambda p, mfld_cfg, *func_args: f(
            p, q, mfld_cfg, *func_args
        )
        gs_wrapped: List[OptimFunc] = [
            lambda p, mfld_cfg, *func_args: g(p, q, mfld_cfg, *func_args) for g in gs
        ]
        hs_wrapped: List[OptimFunc] = [
            lambda p, mfld_cfg, *func_args: h[p, q, mfld_cfg, *func_args] for h in hs
        ]

        # performs constrained optimization (using chosen solver)

        result: ConstrainedSolverResult = method(
            f_wrapped, gs_wrapped, hs_wrapped, p0, optim_mfld_cfg, solve_cfg, *func_args
        )
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

        ctx.save_for_backward(p0)
        ctx.save_for_backward(q)
        ctx.save_for_backward(_v)
        ctx.save_for_backward(p_optimal)

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
        s = len(gs)

        # note that during optimization then the state coordinate on the other
        # manifold does not change during optimization so we reuse it

        conn_coeffs_p = prod_mfld_conn(p, q)
        conn_coeffs_p_optimal = prod_mfld_conn(p_optimal, q)

        r_p = torch.cat(p, q)
        r_p_optimal = torch.cat(p_optimal, q)

        # compute the velocity at the initial point (note that we're expecting
        # the velocity on the components corresponding to the external manifold
        # to be zero as the point remains the same)
        # NOTE: may not be identically zero due to the usage of approximate
        # logarithmic maps being employed in this framework

        # NOTE: uses the same log method as on the optimal manifold but uses
        # the connection associated with the product manifold
        w_p = optim_mfld_cfg.log_method(r_p, r_p_optimal, prod_mfld_conn)

        # we need the inverse metric on the optimizing manifold
        g_inv_p_optimal = optim_mfld_cfg.metric_field(p_optimal).inv()

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
            partial_f, conn_coeffs_p_optimal[:, :n, :], (0, 0)
        )
        for i in range(s):
            partial_g = partial_gs[i]
            hessian_g = hessian_gs[i]

            soln_map_dual += (
                -g_mults[i] / g_vals[i] * torch.outer(partial_g[:n], partial_g)
                + g_mults[i] * hessian_g[:n, :]
                - torch.tensordot(partial_g, conn_coeffs_p_optimal[:, :n, :])
            )

        # as the rightmost side will be multiplied by the tangent vector on the
        # optimization manifold then we slice it but keep the first argument at
        # the full dimensionality of the product manifold (to be compatible
        # with the solution map dual)
        parallel_transp = -torch.tensordot(conn_coeffs_p[:, :, :n], w_p, (1, 0))

        soln_map_jacob = torch.tensordot(
            g_inv_p_optimal,
            torch.tensordot(soln_map_dual, parallel_transp, (1, 0)),
            (1, 0),
        )

        return grad_output * soln_map_jacob


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
        _v: torch.Tensor = None,
        *func_args: *FuncArgs,
    ):
        # NOTE: at time of writing the velocity on the external manifold _v is
        # currently unused and is therefore left undefined (but keeping as part
        # of the API for convenience)

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
        )
        return p_optimal
