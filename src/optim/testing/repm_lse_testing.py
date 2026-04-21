import torch
import numpy as np

from diff_mfld.geometry.metric import RnMetricField
from diff_mfld.mfld import ComputeMfld, Mfld
from optim.constrained.ralm import RalmCfg
from optim.constrained.repm import RepmCfg, repm, RepmMode
from src.optim.subsolvers.rgd import (
    riem_grad_descent,
    RiemGradDescentCfg,
    RiemGradDescentHistory,
    RiemGradDescentResult
)

from optim.constrained.ralm import ralm

from problem import create_problem, TargetCost

target = torch.tensor([2.0, 3.0])
constr_regions = [[(torch.tensor([1.0, 2.0]), 0.5)]]

cost_fn, region_fns = create_problem(target, constr_regions)
region_fn = region_fns[0]

print(type(cost_fn))
print(cost_fn.target)

# sets up the manifold space
metric = RnMetricField(2)
conn = metric.christoffels()
mfld = Mfld(metric, conn)
compute_mfld = ComputeMfld(mfld)

# sets up and runs optimization
cfg = RepmCfg()
p0 = torch.tensor([0.0, 0.0])
result = repm(
    cost_fn, [region_fn], [], p0, RepmMode.LSE, compute_mfld, cfg, ()
)

print(f"success: {result.success}")
print("p_hist:")
print(result.history.p_hist)
print()
print("f_hist:")
print(result.history.f_hist)
