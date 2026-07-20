from enum import Enum

from dmol.diff_mfld.field.util import coord_repr
import torch

# cost metrics


def euler_metric(u_f: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    return torch.eye(2)


def growth_metric(u_f: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    return coord_repr(
        [
            [1.0 + u_f**2, 0.0],
            [0.0, 1.0 + u_t**2],
        ]
    )


def coupled_metric(u_f: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
    return coord_repr(
        [
            [1.0 + u_f**2, u_f * u_t],
            [u_t * u_f, 1.0 + u_t**2],
        ]
    )


class MetricOption(Enum):
    EULER = (euler_metric,)
    GROWTH = (growth_metric,)
    COUPLED = (coupled_metric,)

    def __call__(self, u_f: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        (method,) = self.value
        return method(u_f, u_t)
