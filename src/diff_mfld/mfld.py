import torch
from dataclasses import dataclass
from typing import Optional

from diff_mfld.geometry.metric import MetricField
from diff_mfld.geometry.connection import Connection

from diff_mfld.geodesic.geodesic_funcs import (
    ExpMethod,
    LogMethod,
)

@dataclass
class Mfld:
    metric: Optional[MetricField]
    conn: Connection


@dataclass
class ComputeMfld:
    mfld: Mfld

    exp_method: ExpMethod = ExpMethod.APPROX_O2
    log_method: LogMethod = LogMethod.APPROX_O2
    dist_method: LogMethod = LogMethod.APPROX_O2

    def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.exp_method(p, v, self.mfld.conn)

    def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.log_method(p, q, self.mfld.conn)

    def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        v = self.log_method(p, q, self.mfld.conn)
        metric = self.mfld.metric(p)
        return metric(v, v)

