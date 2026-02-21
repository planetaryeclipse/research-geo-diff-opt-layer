import torch

from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVarTuple

from diff_mfld_optim.mfld_util import MfldCfg
from diff_mfld_optim.geodesic.geodesic_funcs import dist_map

from torch.func import jacfwd, jacrev
from torch.autograd.functional import jacobian

FuncArgs = TypeVarTuple("FuncArgs")
OptimFunc = Callable[[torch.Tensor, MfldCfg, *FuncArgs], torch.Tensor]


@dataclass
class SolverCfg:
    conv_eps = 1e-6
    damp = 0.6
    damp_growth = 0.95  # decays (helps with eventual convergence)
    max_iters = 1000


@dataclass
class SolverResult:
    success: bool
    iters: int
    p: torch.tensor
    p0: torch.tensor


def riem_grad_descent(
    f: OptimFunc,
    p0: torch.Tensor,
    mfld_cfg: MfldCfg,
    solv_cfg: SolverCfg,
    *args: *FuncArgs,
) -> SolverResult:
    # standard optimization algorithm (for us this will act as one of the
    # available subsolvers to be used by ralm)

    p_prev = None
    p: torch.Tensor = p0.detach().clone()

    for i in range(solv_cfg.max_iters):
        # the jacobian takes too long so we abuse backward propagation here to
        # compute the differential of f (the gradient according to torch is
        # equivalent to differential in differential geometric terms)

        # print(f"subsolver: i={i}, p={p}")

        df = jacrev(lambda p: f(p, mfld_cfg, *args))(p)

        # updates the point using the exponential map
        grad_f = mfld_cfg.metric_field(p).sharp(df)
        p = mfld_cfg.exp_method(p, -solv_cfg.damp * grad_f, mfld_cfg.conn)

        if (
            p_prev is not None
            and dist_map(
                p,
                p_prev,
                mfld_cfg.metric_field,
                mfld_cfg.conn,
                mfld_cfg.dist_method,
            )
            <= solv_cfg.conv_eps
        ):
            return SolverResult(True, i + 1, p, p0)

        p_prev = p.clone()  # otherwise it p == p_prev

        if solv_cfg.damp_growth is not None:
            solv_cfg.damp *= solv_cfg.damp_growth

    return SolverResult(False, solv_cfg.max_iters, p, p0)


class SubsolverMethod(Enum):
    RIEM_GRAD_DESCENT = riem_grad_descent

    def __call__(
        self,
        f: OptimFunc,
        p0: torch.Tensor,
        mfld_cfg: MfldCfg,
        solve_cfg: SolverCfg,
        *args: *FuncArgs,
    ):
        self.value(f, p0, mfld_cfg, solve_cfg, *args)
