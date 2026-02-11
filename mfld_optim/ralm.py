import numpy as np
from typing import Callable, Tuple

import torch
from torch.autograd.functional import jacobian

from geodesic_funcs import ExpMethod, LogMethod, DistSquaredMap
from metric import MetricField, Metric, MetricView, RnMetricField
from connection import Connection

from dataclasses import dataclass

# accepts a position on the manifold then outputs the value and differential
Optim_Fn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class MfldCfg:
    metric_field: MetricField
    conn: Connection

    exp_method: ExpMethod = ExpMethod.APPROX_SO
    log_method: LogMethod = LogMethod.APPROX_SO
    dist_method: LogMethod = LogMethod.APPROX_SO


def cost(p, cfg: MfldCfg, q):
    return 0.5 * DistSquaredMap.apply(p, q, cfg.metric_field, cfg.conn, cfg.dist_method)


p = torch.tensor([1.0, 2.0])
q = torch.tensor([4.0, -1.0])
g = RnMetricField(2)

cfg = MfldCfg(g, g.christoffels())

f = cost(p, cfg, q)
df = jacobian(lambda p: cost(p, cfg, q), p, create_graph=True)

print(f)
print(df)
