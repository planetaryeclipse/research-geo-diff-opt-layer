import torch

from diff_mfld.geodesic.geodesic_funcs import ExpMethod, LogMethod
from diff_mfld.geometry.metric import MetricField

def noneuclidean_metric(x1, x2):
    # elements are assigned to this metric to preserve gradient history
    metric = torch.zeros((2, 2))
    metric[0, 0] = 0.5 * x1**2 + 1.0  # added 1. to be nondegenerate at origin
    metric[1, 1] = 0.5 * x2**2 + 1.0
    # metric[2, 2] = 0.5 * x3**2
    return metric

exp_method, log_method = ExpMethod.IVP, LogMethod.BVP

metric_field = MetricField(noneuclidean_metric)
conn = metric_field.christoffels()

p = torch.tensor([1., 2.])
q = torch.tensor([4., -3.])
v_euclid = q - p  # result should not be this

v_log = log_method(p, q, conn)
print(f"v_log = {v_log}")
print(f"v_euclid = {v_euclid}")

q_exp = exp_method(p, v_log, conn)
print(f"q_exp = {q_exp}")

# assert not v_euclid == approx(v_log)