import torch
from time import time

from torch.func import jacrev

from diff_mfld_optim.geometry.metric import RnMetricField
from diff_mfld_optim.optim.subsolver import (
    SolverCfg,
    riem_grad_descent,
    SubsolverMethod,
)
from diff_mfld_optim.optim.constrained import ConstrainedSolverCfg, ralm
from diff_mfld_optim.mfld_util import MfldCfg, dist_squared_map


def test_riem_grad_descent():
    p = torch.tensor([1.0, 2.0])
    q = torch.tensor([4.0, -1.0])

    def f(p, cfg: MfldCfg, q):
        return 0.5 * dist_squared_map(p, q, cfg)

    g = RnMetricField(2)
    mfld_cfg = MfldCfg(g, g.christoffels())
    solv_cfg = SolverCfg()

    result = riem_grad_descent(f, p, mfld_cfg, solv_cfg, q)

    print(f"riem result: {result}")


def test_ralm():
    # NOTE: from testing it seems that generally we want the penalty growth
    # to be larger than the decay rate of the subsolver (seems to sometimes
    # get stuck at a position that violates constraints if we decay too quickly)

    p = torch.tensor([-2.0, 0.0])
    q = torch.tensor([0.0, 0.0])  # want to approach this point

    c = torch.tensor([2.0, 0.0])  # circle centerd on opposite side of p
    rad = 1.0

    def f(p, cfg: MfldCfg, q, c, rad):
        return 0.5 * dist_squared_map(p, q, cfg)

    def g(p, cfg: MfldCfg, q, c, rad):
        # all g must be defined such that g(p) <= 0
        return dist_squared_map(p, c, cfg) - rad**2

    metric_field = RnMetricField(2)
    conn = metric_field.christoffels()

    # warm start the functions
    # print(metric_field(p))
    # print(conn(p))

    mfld_cfg = MfldCfg(metric_field, conn)
    constr_solv_cfg = ConstrainedSolverCfg(
        SubsolverMethod.RIEM_GRAD_DESCENT, SolverCfg()
    )

    print(mfld_cfg.log_method(p, p, mfld_cfg.conn))
    print(dist_squared_map(p, p, mfld_cfg))

    diff = jacrev(lambda q: dist_squared_map(q, p, mfld_cfg))(p)
    print(diff)

    return

    # TODO: improve this optimization time (likely recomputing values so can
    # likely implement caching somewhere in the architecture)

    start_time = time()
    result = ralm(f, [g], [], p, mfld_cfg, constr_solv_cfg, q, c, rad)
    end_time = time()

    print(f"ralm result: {result}")
    print(f"time: {end_time - start_time}")


if __name__ == "__main__":
    # test_riem_grad_descent()
    test_ralm()
