import torch
from diff_mfld_optim.mfld_util import MfldCfg, dist_squared_map

# NOTE: these are the methods necessary to implement the keep-out region for
# the geometric formulation of the training


def cbf_ko(
    p: torch.Tensor,  # current unicycle state
    u: torch.Tensor,  # current input
    ko: torch.Tensor,  # keep-out coniguration
    k_1: torch.Tensor,
    k_2: torch.Tensor,
):
    # evaluates the ghocbf for our system using pytorch (so that it is
    # can be differentiated using autograd to be used in a constrained
    # optimization solver)

    # NOTE: this is not intended as batch evaluation as it is intended to be
    # wrapped as a function that is passed into the

    # for clarity expand the above elements (function signature kept in a more
    # convenient grouping rather than each individual element)
    (x, y, theta, v, omega) = (p[0], p[1], p[2], p[3], p[4])
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
        ko[0],
        ko[1],
        ko[2],
        ko[3],
        ko[4],
        ko[5],
        ko[6],
        ko[7],
        ko[8],
    )

    u_f, _u_t = u[0], u[1]

    # for details refer to `dyn_ext_unicycle_ko_cbf_derivations.ipynb`
    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err ** (7.0 / 2) * x_err
    r_2 = r_sqr_err ** (7.0 / 2) * y_err
    r_3 = r_sqr_err ** (5.0 / 2) * x_err**2
    r_4 = r_sqr_err ** (5.0 / 2) * y_err**2
    r_5 = r_sqr_err ** (5.0 / 2) * x_err * y_err

    # for clarity we split the full condition into a linear system by finding a
    # leading coefficient for the u_f term and a constant
    a_coeff = -(r_1 * torch.cos(theta) + r_2 * torch.sin(theta)) / r_sqr_err**4 * u_f
    b_term = -(
        2 * k_1 * k_2 * r_sqr_err ** (9.0 / 2)
        + 2 * k_1 * k_2 * r_sqr_err**4 * ko_radius
        - 2 * k_2 * r_1 * ko_vel_x
        - 2 * k_2 * r_2 * ko_vel_y
        + 2 * k_2 * r_sqr_err**4 * ko_vel_radius
        - 2 * r_1 * torch.sin(theta) * v * omega
        - 2 * r_1 * ko_accel_x
        + 2 * r_2 * torch.cos(theta) * v * omega
        - 2 * r_2 * ko_accel_y
        + r_3 * torch.cos(2 * theta) * v**2
        + r_3 * v**3
        + 2 * r_3 * ko_accel_x**2
        - r_4 * torch.cos(2 * theta) * v**2
        + r_4 * v**2
        + 2 * r_4 * ko_accel_y**2
        + 2 * r_5 * torch.sin(2 * theta) * v**2
        + 4 * r_5 * ko_vel_x * ko_vel_y
        - 2 * r_sqr_err ** (7.0 / 2) * v**2
        - 2 * r_sqr_err ** (7.0 / 2) * ko_vel_x ** 2
        - 2 * r_sqr_err ** (7.0 / 2) * ko_vel_y ** 2
        + 2 * r_sqr_err**4 * ko_accel_radius
    ) / (2 * r_sqr_err**4)

    return a_coeff + b_term


def gcbf_f(u: torch.Tensor, mfld_cfg: MfldCfg, u_nom: torch.Tensor):
    # minimization of correction between nominal control input and safe input
    return 0.5 * dist_squared_map(u, u_nom, mfld_cfg)  # >= 0
