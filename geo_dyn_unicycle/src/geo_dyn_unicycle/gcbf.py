from dmol.diff_mfld.bundle.vector_bundle import ScalarBundle
from dmol.diff_mfld.connection.methods.geod_log_diff import LogMapCovarMethod
from dmol.diff_mfld.connection.methods.methods import LogMapMethod
from dmol.diff_mfld.field.field_types import LambdaField, ScalarField
from dmol.diff_mfld.field.riem_fields import RiemSqrDist
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold, Point
from dmol.diff_mfld.riemann import MetricField
import torch


def gcbf_cost[U: Manifold](
    u_nom: Point[U],
    metric: MetricField[U],
    log_map_method: LogMapMethod,
    log_map_covar_method: LogMapCovarMethod,
) -> ScalarField[U]:
    return 0.5 * RiemSqrDist[U](u_nom, metric, log_map_method, log_map_covar_method)


def gcbf_constr[U: Manifold](
    input_space: type[U],  # input space type
    uni_state: torch.Tensor,
    keep_out: torch.Tensor,
    k1: float,
    k2: float,
) -> ScalarField[U]:

    # as the cbf condition is defined independently on the configuration manifold and is therefore constant during
    # optimization which generates the simple scalar field below

    x, y, theta, v, omega = (
        uni_state[0],
        uni_state[1],
        uni_state[2],
        uni_state[3],
        uni_state[4],
    )
    (
        ko_x,
        ko_y,
        ko_vel_x,
        ko_vel_y,
        ko_accel_x,
        ko_accel_y,
        ko_radius,
        ko_vel_radius,
        ko_accel_radius,
    ) = (
        keep_out[0],
        keep_out[1],
        keep_out[2],
        keep_out[3],
        keep_out[4],
        keep_out[5],
        keep_out[6],
        keep_out[7],
        keep_out[8],
    )

    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err ** (7.0 / 2) * x_err
    r_2 = r_sqr_err ** (7.0 / 2) * y_err
    r_3 = r_sqr_err ** (5.0 / 2) * x_err**2
    r_4 = r_sqr_err ** (5.0 / 2) * y_err**2
    r_5 = r_sqr_err ** (5.0 / 2) * x_err * y_err

    # find a leading coefficient for the forward input and a constant
    a_coeff = -(+2 * r_1 * torch.cos(theta) + 2 * r_2 * torch.sin(theta)) / (
        2 * r_sqr_err**4
    )
    b_term = -(
        -2 * k1 * k2 * r_sqr_err ** (9 / 2)
        + 2 * k1 * k2 * r_sqr_err**4 * ko_radius
        + 2 * k2 * r_sqr_err**4 * ko_vel_radius
        - 2 * k2 * r_1 * ko_vel_x
        - 2 * k2 * r_2 * ko_vel_y
        - 2 * r_sqr_err ** (7 / 2) * v**2
        - 2 * r_sqr_err ** (7 / 2) * ko_vel_x**2
        - 2 * r_sqr_err ** (7 / 2) * ko_vel_y**2
        + 2 * r_sqr_err**4 * ko_accel_radius
        - 2 * r_1 * torch.sin(theta) * v * omega
        - 2 * r_1 * ko_accel_x
        + 2 * r_2 * torch.cos(theta) * v * omega
        - 2 * r_2 * ko_accel_y
        + r_3 * torch.cos(theta) * v**2
        + r_3 * v**2
        + 2 * r_3 * ko_vel_x**2
        - r_4 * torch.cos(theta) * v**2
        + r_4 * v**2
        + 2 * r_4 * ko_vel_y**2
        + 2 * r_5 * torch.sin(theta) * v**2
        + 4 * r_5 * ko_vel_x * ko_vel_y
    ) / (2 * r_sqr_err**4)

    cbf_field = LambdaField[ScalarBundle[input_space]](
        lambda u_f, _u_t: coord_repr(a_coeff * u_f + b_term)  # type: ignore
    )
    return cbf_field  # type: ignore
