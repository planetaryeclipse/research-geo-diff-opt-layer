import torch
import numpy as np

from diff_mfld.geometry.metric import RnMetricField
from diff_mfld.mfld import ComputeMfld, Mfld
from src.optim.subsolvers.rgd import (
    riem_grad_descent,
    RiemGradDescentCfg,
    RiemGradDescentHistory,
    RiemGradDescentResult
)

from problem import create_problem, TargetCost

target = torch.tensor([2.0, 3.0])
cost_fn, _ = create_problem(target)

print(type(cost_fn))
print(cost_fn.target)

# sets up the manifold space
metric = RnMetricField(2)
conn = metric.christoffels()
mfld = Mfld(metric, conn)
compute_mfld = ComputeMfld(mfld)

# sets up and runs optimization
cfg = RiemGradDescentCfg()
p0 = torch.tensor([0.0, 0.0])
result = riem_grad_descent(cost_fn, p0, compute_mfld, cfg, ())

print(f"success: {result.success}")
print("p_hist:")
print(result.history.p_hist)
print()
print("f_hist:")
print(result.history.f_hist)
