import torch
from dataclasses import dataclass

from diff_mfld_optim.geometry.metric import MetricField
from diff_mfld_optim.geometry.connection import Connection

from diff_mfld_optim.geodesic.geodesic_funcs import (
    ExpMethod,
    LogMethod,
)

import diff_mfld_optim.geodesic.geodesic_funcs as geodesic_funcs


@dataclass
class MfldCfg:
    metric_field: MetricField
    conn: Connection

    exp_method: ExpMethod = ExpMethod.APPROX_SO
    log_method: LogMethod = LogMethod.APPROX_SO
    dist_method: LogMethod = LogMethod.APPROX_SO


# utility functions that depend on mfld cfg


def dist_squared_map(p, q, cfg: MfldCfg) -> torch.tensor:
    # allows clean mfldcfg interface for use in cost and constraint functions
    return geodesic_funcs.dist_squared_map(
        p, q, cfg.metric_field, cfg.conn, cfg.dist_method
    )
