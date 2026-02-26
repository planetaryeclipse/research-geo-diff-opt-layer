import torch
from diff_mfld_optim.mfld_util import MfldCfg, dist_squared_map


from diff_mfld_optim.geometry.funcs import (
    MfldFunc,
    FuncArgs,
    RiemannSquaredDist,
    ConstrMfldFunc,
)

# these are the methods necessary to implement the keep-out region for the
# non-product geometric formulation of the GHOCBF-QP


def cbf_ko(
    p: torch.Tensor,  # current unicycle state
    u: torch.Tensor,  # current input
    ko: torch.Tensor,  # keep-out coniguration
    k1: torch.Tensor,  # cbf param
    k2: torch.Tensor,  # cbf param
):
    # NOTE: this function remains unchanged by going from the non-product to
    # product manifold configurations

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
    u_f = u[0]

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
    return a_coeff * u_f + b_term


def cbf_ko_diff(
    p: torch.Tensor,  # current unicycle state
    u: torch.Tensor,  # current input
    ko: torch.Tensor,  # keep-out coniguration
    k1: torch.Tensor,  # cbf param
    k2: torch.Tensor,  # cbf param
):
    (x, y, theta, v, omega) = (p[0], p[1], p[2], p[3], p[4])
    (
        ko_x,
        ko_y,
        ko_vel_x,
        ko_vel_y,
    ) = (
        ko[0],
        ko[1],
        ko[2],
        ko[3],
    )
    u_f = u[0]

    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err * x**2
    r_2 = r_sqr_err * y**2
    r_3 = r_sqr_err * x * y
    r_4 = x**2 * y
    r_5 = x * y**2

    r_6 = ko_x**2
    r_7 = ko_y**2
    r_8 = ko_x * ko_y

    r_9 = x * ko_y
    r_10 = y * ko_x

    n = p.shape[0]
    m = u.shape[0]

    grad = torch.zeros((n + m))

    # NOTE: generated via sympy

    # these first two components are the gradient with respect to the control
    # coordinates (and the last 5 are associated with the system coords)
    grad[0] = (
        (x - ko_x) * torch.cos(theta) + (y - ko_y) * torch.sin(theta)
    ) / torch.sqrt(r_sqr_err)
    grad[1] = 0.0

    grad[2] = (
        k1 * k2 * r_sqr_err**2 * x
        - k1 * k2 * r_sqr_err**2 * ko_x
        - k2 * r_sqr_err**2 * ko_vel_x
        - k2 * r_sqr_err * r_9 * ko_vel_y
        - k2 * r_sqr_err * r_10 * ko_vel_y
        + k2 * r_sqr_err * r_6 * ko_vel_x
        + k2 * r_sqr_err * r_8 * ko_vel_y
        - 2 * k2 * r_sqr_err * x * ko_x * ko_vel_x
        + k2 * r_1 * ko_vel_x
        + k2 * r_3 * ko_vel_y
        - r_sqr_err**2 * ko_vel_x
        - r_sqr_err * r_9 * ko_vel_y
        - r_sqr_err * r_10 * ko_vel_y
        + r_sqr_err * r_6 * ko_vel_x
        + r_sqr_err * r_8 * ko_vel_y
        - 2 * r_sqr_err * x * ko_x * ko_vel_x
        - 3 * r_sqr_err * x * ko_vel_x**2
        - r_sqr_err * x * ko_vel_y**2
        - 2 * r_sqr_err * y * ko_vel_x * ko_vel_y
        + 3 * r_sqr_err * ko_x * ko_vel_x**2
        + r_sqr_err * ko_x * ko_vel_y**2
        + 2 * r_sqr_err * ko_y * ko_vel_x * ko_vel_y
        + r_1 * ko_vel_x
        + r_3 * ko_vel_y
        + v**2
        * (
            9 * r_6 * x
            + 3 * x**3
            - 9 * x**2 * ko_x
            - 3 * x * x_err**2
            - 3 * x * y_err**2
            + 3 * x_err**2 * ko_x
            + 3 * y_err**2 * ko_x
            - 3 * ko_x**3
        )
        * torch.cos(theta) ** 2
        + v**2
        * (
            2 * r_sqr_err * x
            - 2 * r_sqr_err * ko_x
            - 6 * r_9 * y
            - 3 * r_10 * y
            + 3 * r_5
            + 3 * r_7 * x
            - 3 * r_7 * ko_x
            + 6 * r_8 * y
            - 3 * x * x_err**2
            - 3 * x * y_err**2
            + 3 * x_err**2 * ko_x
            + 3 * y_err**2 * ko_x
        )
        * torch.sin(theta) ** 2
        + v
        * (
            omega * r_sqr_err * r_9
            + omega * r_sqr_err * r_10
            - omega * r_sqr_err * r_8
            - omega * r_3
        )
        * torch.cos(theta)
        + v
        * (
            -omega * r_sqr_err**2
            + omega * r_sqr_err * r_6
            - 2 * omega * r_sqr_err * x * ko_x
            + omega * r_1
            - 2 * r_sqr_err * v * y * torch.cos(theta)
            + 2 * r_sqr_err * v * ko_y * torch.cos(theta)
            - 6 * r_9 * v * x * torch.cos(theta)
            - 12 * r_10 * v * x * torch.cos(theta)
            + 6 * r_4 * v * torch.cos(theta)
            + 6 * r_6 * v * y * torch.cos(theta)
            - 6 * r_6 * v * ko_y * torch.cos(theta)
            + 12 * r_8 * v * x * torch.cos(theta)
        )
        * torch.sin(theta)
        + 3
        * x
        * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y) ** 2
        + (
            r_sqr_err**2 * u_f
            - r_sqr_err * r_6 * u_f
            + 2 * r_sqr_err * u_f * x * ko_x
            - r_1 * u_f
        )
        * torch.cos(theta)
        - 3
        * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y) ** 2
        * ko_x
        + (
            r_sqr_err * r_9 * u_f
            + r_sqr_err * r_10 * u_f
            - r_sqr_err * r_8 * u_f
            - r_3 * u_f
        )
        * torch.sin(theta)
    ) / r_sqr_err ** (5 / 2)
    grad[3] = (
        k1 * k2 * r_sqr_err**2 * y
        - k1 * k2 * r_sqr_err**2 * ko_y
        - k2 * r_sqr_err**2 * ko_vel_y
        - k2 * r_sqr_err * r_9 * ko_vel_x
        - k2 * r_sqr_err * r_10 * ko_vel_x
        + k2 * r_sqr_err * r_7 * ko_vel_y
        + k2 * r_sqr_err * r_8 * ko_vel_x
        - 2 * k2 * r_sqr_err * y * ko_y * ko_vel_y
        + k2 * r_2 * ko_vel_y
        + k2 * r_3 * ko_vel_x
        - r_sqr_err**2 * ko_vel_y
        - r_sqr_err * r_9 * ko_vel_x
        - r_sqr_err * r_10 * ko_vel_x
        + r_sqr_err * r_7 * ko_vel_y
        + r_sqr_err * r_8 * ko_vel_x
        - 2 * r_sqr_err * x * ko_vel_x * ko_vel_y
        - 2 * r_sqr_err * y * ko_y * ko_vel_y
        - r_sqr_err * y * ko_vel_x**2
        - 3 * r_sqr_err * y * ko_vel_y**2
        + 2 * r_sqr_err * ko_x * ko_vel_x * ko_vel_y
        + r_sqr_err * ko_y * ko_vel_x**2
        + 3 * r_sqr_err * ko_y * ko_vel_y**2
        + r_2 * ko_vel_y
        + r_3 * ko_vel_x
        + v**2
        * (
            9 * r_7 * y
            - 3 * x_err**2 * y
            + 3 * x_err**2 * ko_y
            + 3 * y**3
            - 9 * y**2 * ko_y
            - 3 * y * y_err**2
            + 3 * y_err**2 * ko_y
            - 3 * ko_y**3
        )
        * torch.sin(theta) ** 2
        + v**2
        * (
            2 * r_sqr_err * y
            - 2 * r_sqr_err * ko_y
            - 3 * r_9 * x
            - 6 * r_10 * x
            + 3 * r_4
            + 3 * r_6 * y
            - 3 * r_6 * ko_y
            + 6 * r_8 * x
            - 3 * x_err**2 * y
            + 3 * x_err**2 * ko_y
            - 3 * y * y_err**2
            + 3 * y_err**2 * ko_y
        )
        * torch.cos(theta) ** 2
        + v
        * (
            omega * r_sqr_err**2
            - omega * r_sqr_err * r_7
            + 2 * omega * r_sqr_err * y * ko_y
            - omega * r_2
        )
        * torch.cos(theta)
        + v
        * (
            -omega * r_sqr_err * r_9
            - omega * r_sqr_err * r_10
            + omega * r_sqr_err * r_8
            + omega * r_3
            - 2 * r_sqr_err * v * x * torch.cos(theta)
            + 2 * r_sqr_err * v * ko_x * torch.cos(theta)
            - 12 * r_9 * v * y * torch.cos(theta)
            - 6 * r_10 * v * y * torch.cos(theta)
            + 6 * r_5 * v * torch.cos(theta)
            + 6 * r_7 * v * x * torch.cos(theta)
            - 6 * r_7 * v * ko_x * torch.cos(theta)
            + 12 * r_8 * v * y * torch.cos(theta)
        )
        * torch.sin(theta)
        + 3
        * y
        * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y) ** 2
        + (
            r_sqr_err**2 * u_f
            - r_sqr_err * r_7 * u_f
            + 2 * r_sqr_err * u_f * y * ko_y
            - r_2 * u_f
        )
        * torch.sin(theta)
        - 3
        * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y) ** 2
        * ko_y
        + (
            r_sqr_err * r_9 * u_f
            + r_sqr_err * r_10 * u_f
            - r_sqr_err * r_8 * u_f
            - r_3 * u_f
        )
        * torch.cos(theta)
    ) / r_sqr_err ** (5 / 2)
    grad[4] = (
        v**2 * (-2 * r_9 - 2 * r_10 + 2 * r_8 + 2 * x * y) * torch.sin(theta) ** 2
        + v**2 * (2 * r_9 + 2 * r_10 - 2 * r_8 - 2 * x * y) * torch.cos(theta) ** 2
        + v * (-omega * r_sqr_err * x + omega * r_sqr_err * ko_x) * torch.cos(theta)
        + v
        * (
            -omega * r_sqr_err * y
            + omega * r_sqr_err * ko_y
            + 2 * r_6 * v * torch.cos(theta)
            - 2 * r_7 * v * torch.cos(theta)
            + 2 * v * x**2 * torch.cos(theta)
            - 4 * v * x * ko_x * torch.cos(theta)
            - 2 * v * y**2 * torch.cos(theta)
            + 4 * v * y * ko_y * torch.cos(theta)
        )
        * torch.sin(theta)
        + (-r_sqr_err * u_f * x + r_sqr_err * u_f * ko_x) * torch.sin(theta)
        + (r_sqr_err * u_f * y - r_sqr_err * u_f * ko_y) * torch.cos(theta)
    ) / r_sqr_err ** (3 / 2)
    grad[5] = (
        v
        * (
            -2 * r_6 * torch.cos(theta)
            - 2 * x**2 * torch.cos(theta)
            + 4 * x * ko_x * torch.cos(theta)
            + 2 * x_err**2 * torch.cos(theta)
            + 2 * y_err**2 * torch.cos(theta)
        )
        * torch.cos(theta)
        + v
        * (
            4 * r_9 * torch.cos(theta)
            + 4 * r_10 * torch.cos(theta)
            - 2 * r_7 * torch.sin(theta)
            - 4 * r_8 * torch.cos(theta)
            - 4 * x * y * torch.cos(theta)
            + 2 * x_err**2 * torch.sin(theta)
            - 2 * y**2 * torch.sin(theta)
            + 4 * y * ko_y * torch.sin(theta)
            + 2 * y_err**2 * torch.sin(theta)
        )
        * torch.sin(theta)
        + (-omega * r_sqr_err * x + omega * r_sqr_err * ko_x) * torch.sin(theta)
        + (omega * r_sqr_err * y - omega * r_sqr_err * ko_y) * torch.cos(theta)
    ) / r_sqr_err ** (3 / 2)
    grad[6] = (
        v
        * ((-x + ko_x) * torch.sin(theta) + (y - ko_y) * torch.cos(theta))
        / torch.sqrt(r_sqr_err)
    )

    return grad


def cbf_ko_hess(
    p,
    u: torch.Tensor,
    ko: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cfg: MfldCfg,
):
    (x, y, theta, v, omega) = (p[0], p[1], p[2], p[3], p[4])
    (
        ko_x,
        ko_y,
        ko_vel_x,
        ko_vel_y,
    ) = (
        ko[0],
        ko[1],
        ko[2],
        ko[3],
    )
    u_f = u[0]

    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err * x**2
    r_2 = r_sqr_err * y**2
    r_3 = r_sqr_err * x * y
    r_4 = x**2 * y
    r_5 = x * y**2

    r_6 = ko_x**2
    r_7 = ko_y**2
    r_8 = ko_x * ko_y
    r_9 = ko_x**2 * ko_y
    r_10 = ko_x * ko_y**2

    r_11 = x * ko_y
    r_12 = y * ko_x
    r_13 = x * ko_y**2
    r_14 = y * ko_x**2
    r_15 = ko_x**2 * ko_y**2

    n = p.shape[0]
    m = u.shape[0]

    # NOTE: sympy generated code (with manual symbol correction) was used to
    # generate the expressions to compute the upper right half of the
    # coefficient hessian (to avoid computational cost)
    coeff_hessian = torch.zeros((n + m, n + m))
    coeff_hessian[0, :] = (
        [
            0,
            0,
            (
                (r_sqr_err - r_6 - x**2 + 2 * x * ko_x) * torch.cos(theta)
                + (-x * y + x * ko_y + y * ko_x - ko_x * ko_y) * torch.sin(theta)
            )
            / r_sqr_err ** (3 / 2),
            (
                (r_sqr_err - y**2 + 2 * y * ko_y - ko_y**2) * torch.sin(theta)
                + (-x * y + x * ko_y + y * ko_x - ko_x * ko_y) * torch.cos(theta)
            )
            / r_sqr_err ** (3 / 2),
            ((-x + ko_x) * torch.sin(theta) + (y - ko_y) * torch.cos(theta))
            / torch.sqrt(r_sqr_err),
            0,
            0,
        ],
    )

    coeff_hessian[1, :] = ([0, 0, 0, 0, 0, 0, 0],)
    coeff_hessian[2, :] = (
        [
            0,
            0,
            (
                k1 * k2 * r_sqr_err**3
                - 2 * k1 * k2 * r_sqr_err**2 * r_6
                + 4 * k1 * k2 * r_sqr_err**2 * x * ko_x
                + k1 * k2 * r_sqr_err**2 * x_err**2
                - 2 * k1 * k2 * r_sqr_err * r_1
                + 3 * k2 * r_sqr_err**2 * x * ko_vel_x
                + k2 * r_sqr_err**2 * y * ko_vel_y
                - 3 * k2 * r_sqr_err**2 * ko_x * ko_vel_x
                - k2 * r_sqr_err**2 * ko_y * ko_vel_y
                - 9 * k2 * r_sqr_err * r_6 * x * ko_vel_x
                - 3 * k2 * r_sqr_err * r_6 * y * ko_vel_y
                + 3 * k2 * r_sqr_err * r_6 * ko_y * ko_vel_y
                - 6 * k2 * r_sqr_err * x * ko_x * ko_y * ko_vel_y
                + 3 * k2 * r_sqr_err * ko_x**3 * ko_vel_x
                - 3 * k2 * r_1 * x * ko_vel_x
                - 3 * k2 * r_1 * y * ko_vel_y
                + 9 * k2 * r_1 * ko_x * ko_vel_x
                + 3 * k2 * r_1 * ko_y * ko_vel_y
                + 6 * k2 * r_3 * ko_x * ko_vel_y
                + 3 * r_sqr_err**2 * x * ko_vel_x
                + r_sqr_err**2 * y * ko_vel_y
                - 3 * r_sqr_err**2 * ko_x * ko_vel_x
                - r_sqr_err**2 * ko_y * ko_vel_y
                - 3 * r_sqr_err**2 * ko_vel_x**2
                - r_sqr_err**2 * ko_vel_y**2
                - 9 * r_sqr_err * r_6 * x * ko_vel_x
                - 3 * r_sqr_err * r_6 * y * ko_vel_y
                + 3 * r_sqr_err * r_6 * ko_y * ko_vel_y
                + 15 * r_sqr_err * r_6 * ko_vel_x**2
                + 3 * r_sqr_err * r_6 * ko_vel_y**2
                - 6 * r_sqr_err * x * ko_x * ko_y * ko_vel_y
                - 30 * r_sqr_err * x * ko_x * ko_vel_x**2
                - 6 * r_sqr_err * x * ko_x * ko_vel_y**2
                - 12 * r_sqr_err * x * ko_y * ko_vel_x * ko_vel_y
                - 12 * r_sqr_err * y * ko_x * ko_vel_x * ko_vel_y
                + 3
                * r_sqr_err
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                + 3 * r_sqr_err * ko_x**3 * ko_vel_x
                + 12 * r_sqr_err * ko_x * ko_y * ko_vel_x * ko_vel_y
                - 3 * r_1 * x * ko_vel_x
                - 3 * r_1 * y * ko_vel_y
                + 9 * r_1 * ko_x * ko_vel_x
                + 3 * r_1 * ko_y * ko_vel_y
                + 15 * r_1 * ko_vel_x**2
                + 3 * r_1 * ko_vel_y**2
                + 6 * r_3 * ko_x * ko_vel_y
                + 12 * r_3 * ko_vel_x * ko_vel_y
                - 15
                * r_6
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                + v**2
                * (
                    3 * r_sqr_err * r_6
                    - 6 * r_sqr_err * x * ko_x
                    - 3 * r_sqr_err * x_err**2
                    - 3 * r_sqr_err * y_err**2
                    + 3 * r_1
                    - 15 * r_6**2
                    - 90 * r_6 * x**2
                    + 15 * r_6 * x_err**2
                    + 15 * r_6 * y_err**2
                    - 15 * x**4
                    + 60 * x**3 * ko_x
                    + 15 * x**2 * x_err**2
                    + 15 * x**2 * y_err**2
                    - 30 * x * x_err**2 * ko_x
                    - 30 * x * y_err**2 * ko_x
                    + 60 * x * ko_x**3
                )
                * torch.cos(theta) ** 2
                + v**2
                * (
                    2 * r_sqr_err**2
                    - 12 * r_sqr_err * r_6
                    + 24 * r_sqr_err * x * ko_x
                    - 3 * r_sqr_err * x_err**2
                    - 6 * r_sqr_err * y * ko_y
                    - 3 * r_sqr_err * y_err**2
                    + 3 * r_sqr_err * ko_y**2
                    - 12 * r_1
                    + 3 * r_2
                    - 15 * r_4 * y
                    + 30 * r_4 * ko_y
                    + 30 * r_5 * ko_x
                    + 15 * r_6 * x_err**2
                    - 15 * r_6 * y**2
                    + 30 * r_6 * y * ko_y
                    + 15 * r_6 * y_err**2
                    - 15 * r_6 * ko_y**2
                    + 15 * x**2 * x_err**2
                    + 15 * x**2 * y_err**2
                    - 15 * x**2 * ko_y**2
                    - 30 * x * x_err**2 * ko_x
                    - 60 * x * y * ko_x * ko_y
                    - 30 * x * y_err**2 * ko_x
                    + 30 * x * ko_x * ko_y**2
                )
                * torch.sin(theta) ** 2
                + v
                * (
                    -omega * r_sqr_err**2 * y
                    + omega * r_sqr_err**2 * ko_y
                    + 3 * omega * r_sqr_err * r_6 * y
                    - 3 * omega * r_sqr_err * r_6 * ko_y
                    + 6 * omega * r_sqr_err * x * ko_x * ko_y
                    + 3 * omega * r_1 * y
                    - 3 * omega * r_1 * ko_y
                    - 6 * omega * r_3 * ko_x
                )
                * torch.cos(theta)
                + v
                * (
                    3 * omega * r_sqr_err**2 * x
                    - 3 * omega * r_sqr_err**2 * ko_x
                    - 9 * omega * r_sqr_err * r_6 * x
                    + 3 * omega * r_sqr_err * ko_x**3
                    - 3 * omega * r_1 * x
                    + 9 * omega * r_1 * ko_x
                    - 18 * r_sqr_err * v * x * ko_y * torch.cos(theta)
                    - 18 * r_sqr_err * v * y * ko_x * torch.cos(theta)
                    + 18 * r_sqr_err * v * ko_x * ko_y * torch.cos(theta)
                    + 18 * r_3 * v * torch.cos(theta)
                    - 30 * r_4 * v * x * torch.cos(theta)
                    + 90 * r_4 * v * ko_x * torch.cos(theta)
                    - 90 * r_6 * v * x * y * torch.cos(theta)
                    + 90 * r_6 * v * x * ko_y * torch.cos(theta)
                    + 30 * v * x**3 * ko_y * torch.cos(theta)
                    - 90 * v * x**2 * ko_x * ko_y * torch.cos(theta)
                    + 30 * v * y * ko_x**3 * torch.cos(theta)
                    - 30 * v * ko_x**3 * ko_y * torch.cos(theta)
                )
                * torch.sin(theta)
                - 15
                * x**2
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                + 30
                * x
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                * ko_x
                + (
                    -3 * r_sqr_err**2 * u_f * x
                    + 3 * r_sqr_err**2 * u_f * ko_x
                    + 9 * r_sqr_err * r_6 * u_f * x
                    - 3 * r_sqr_err * u_f * ko_x**3
                    + 3 * r_1 * u_f * x
                    - 9 * r_1 * u_f * ko_x
                )
                * torch.cos(theta)
                + (
                    -(r_sqr_err**2) * u_f * y
                    + r_sqr_err**2 * u_f * ko_y
                    + 3 * r_sqr_err * r_6 * u_f * y
                    - 3 * r_sqr_err * r_6 * u_f * ko_y
                    + 6 * r_sqr_err * u_f * x * ko_x * ko_y
                    + 3 * r_1 * u_f * y
                    - 3 * r_1 * u_f * ko_y
                    - 6 * r_3 * u_f * ko_x
                )
                * torch.sin(theta)
            )
            / r_sqr_err ** (7 / 2),
            (
                k1 * k2 * r_sqr_err**2 * x * ko_y
                + k1 * k2 * r_sqr_err**2 * y * ko_x
                - k1 * k2 * r_sqr_err**2 * ko_x * ko_y
                - k1 * k2 * r_sqr_err * r_3
                + k2 * r_sqr_err**2 * x * ko_vel_y
                + k2 * r_sqr_err**2 * y * ko_vel_x
                - k2 * r_sqr_err**2 * ko_x * ko_vel_y
                - k2 * r_sqr_err**2 * ko_y * ko_vel_x
                - 3 * k2 * r_sqr_err * r_6 * y * ko_vel_x
                + 3 * k2 * r_sqr_err * r_6 * ko_y * ko_vel_x
                - 6 * k2 * r_sqr_err * x * ko_x * ko_y * ko_vel_x
                - 3 * k2 * r_sqr_err * x * ko_y**2 * ko_vel_y
                - 6 * k2 * r_sqr_err * y * ko_x * ko_y * ko_vel_y
                + 3 * k2 * r_sqr_err * ko_x * ko_y**2 * ko_vel_y
                - 3 * k2 * r_1 * y * ko_vel_x
                + 3 * k2 * r_1 * ko_y * ko_vel_x
                - 3 * k2 * r_2 * x * ko_vel_y
                + 3 * k2 * r_2 * ko_x * ko_vel_y
                + 6 * k2 * r_3 * ko_x * ko_vel_x
                + 6 * k2 * r_3 * ko_y * ko_vel_y
                + r_sqr_err**2 * x * ko_vel_y
                + r_sqr_err**2 * y * ko_vel_x
                - r_sqr_err**2 * ko_x * ko_vel_y
                - r_sqr_err**2 * ko_y * ko_vel_x
                - 2 * r_sqr_err**2 * ko_vel_x * ko_vel_y
                - 3 * r_sqr_err * r_6 * y * ko_vel_x
                + 3 * r_sqr_err * r_6 * ko_y * ko_vel_x
                + 6 * r_sqr_err * r_6 * ko_vel_x * ko_vel_y
                - 6 * r_sqr_err * x * ko_x * ko_y * ko_vel_x
                - 12 * r_sqr_err * x * ko_x * ko_vel_x * ko_vel_y
                - 3 * r_sqr_err * x * ko_y**2 * ko_vel_y
                - 9 * r_sqr_err * x * ko_y * ko_vel_x**2
                - 9 * r_sqr_err * x * ko_y * ko_vel_y**2
                - 6 * r_sqr_err * y * ko_x * ko_y * ko_vel_y
                - 9 * r_sqr_err * y * ko_x * ko_vel_x**2
                - 9 * r_sqr_err * y * ko_x * ko_vel_y**2
                - 12 * r_sqr_err * y * ko_y * ko_vel_x * ko_vel_y
                + 3 * r_sqr_err * ko_x * ko_y**2 * ko_vel_y
                + 9 * r_sqr_err * ko_x * ko_y * ko_vel_x**2
                + 9 * r_sqr_err * ko_x * ko_y * ko_vel_y**2
                + 6 * r_sqr_err * ko_y**2 * ko_vel_x * ko_vel_y
                - 3 * r_1 * y * ko_vel_x
                + 3 * r_1 * ko_y * ko_vel_x
                + 6 * r_1 * ko_vel_x * ko_vel_y
                - 3 * r_2 * x * ko_vel_y
                + 3 * r_2 * ko_x * ko_vel_y
                + 6 * r_2 * ko_vel_x * ko_vel_y
                + 6 * r_3 * ko_x * ko_vel_x
                + 6 * r_3 * ko_y * ko_vel_y
                + 9 * r_3 * ko_vel_x**2
                + 9 * r_3 * ko_vel_y**2
                + v**2
                * (
                    6 * r_sqr_err * x * ko_y
                    + 6 * r_sqr_err * y * ko_x
                    - 6 * r_sqr_err * ko_x * ko_y
                    - 6 * r_3
                    - 15 * r_4 * x
                    + 45 * r_4 * ko_x
                    - 45 * r_6 * x * y
                    + 45 * r_6 * x * ko_y
                    + 15 * x**3 * ko_y
                    - 45 * x**2 * ko_x * ko_y
                    + 15 * x * x_err**2 * y
                    - 15 * x * x_err**2 * ko_y
                    + 15 * x * y * y_err**2
                    - 15 * x * y_err**2 * ko_y
                    - 15 * x_err**2 * y * ko_x
                    + 15 * x_err**2 * ko_x * ko_y
                    - 15 * y * y_err**2 * ko_x
                    + 15 * y * ko_x**3
                    + 15 * y_err**2 * ko_x * ko_y
                    - 15 * ko_x**3 * ko_y
                )
                * torch.cos(theta) ** 2
                + v**2
                * (
                    6 * r_sqr_err * x * ko_y
                    + 6 * r_sqr_err * y * ko_x
                    - 6 * r_sqr_err * ko_x * ko_y
                    - 6 * r_3
                    - 15 * r_5 * y
                    + 45 * r_5 * ko_y
                    + 15 * x * x_err**2 * y
                    - 15 * x * x_err**2 * ko_y
                    + 15 * x * y * y_err**2
                    - 45 * x * y * ko_y**2
                    - 15 * x * y_err**2 * ko_y
                    + 15 * x * ko_y**3
                    - 15 * x_err**2 * y * ko_x
                    + 15 * x_err**2 * ko_x * ko_y
                    + 15 * y**3 * ko_x
                    - 45 * y**2 * ko_x * ko_y
                    - 15 * y * y_err**2 * ko_x
                    + 45 * y * ko_x * ko_y**2
                    + 15 * y_err**2 * ko_x * ko_y
                    - 15 * ko_x * ko_y**3
                )
                * torch.sin(theta) ** 2
                + v
                * (
                    -omega * r_sqr_err**2 * x
                    + omega * r_sqr_err**2 * ko_x
                    + 3 * omega * r_sqr_err * x * ko_y**2
                    + 6 * omega * r_sqr_err * y * ko_x * ko_y
                    - 3 * omega * r_sqr_err * ko_x * ko_y**2
                    + 3 * omega * r_2 * x
                    - 3 * omega * r_2 * ko_x
                    - 6 * omega * r_3 * ko_y
                )
                * torch.cos(theta)
                + v
                * (
                    omega * r_sqr_err**2 * y
                    - omega * r_sqr_err**2 * ko_y
                    - 3 * omega * r_sqr_err * r_6 * y
                    + 3 * omega * r_sqr_err * r_6 * ko_y
                    - 6 * omega * r_sqr_err * x * ko_x * ko_y
                    - 3 * omega * r_1 * y
                    + 3 * omega * r_1 * ko_y
                    + 6 * omega * r_3 * ko_x
                    - 2 * r_sqr_err**2 * v * torch.cos(theta)
                    + 6 * r_sqr_err * r_6 * v * torch.cos(theta)
                    - 12 * r_sqr_err * v * x * ko_x * torch.cos(theta)
                    - 12 * r_sqr_err * v * y * ko_y * torch.cos(theta)
                    + 6 * r_sqr_err * v * ko_y**2 * torch.cos(theta)
                    + 6 * r_1 * v * torch.cos(theta)
                    + 6 * r_2 * v * torch.cos(theta)
                    - 30 * r_4 * v * y * torch.cos(theta)
                    + 60 * r_4 * v * ko_y * torch.cos(theta)
                    + 60 * r_5 * v * ko_x * torch.cos(theta)
                    - 30 * r_6 * v * y**2 * torch.cos(theta)
                    + 60 * r_6 * v * y * ko_y * torch.cos(theta)
                    - 30 * r_6 * v * ko_y**2 * torch.cos(theta)
                    - 30 * v * x**2 * ko_y**2 * torch.cos(theta)
                    - 120 * v * x * y * ko_x * ko_y * torch.cos(theta)
                    + 60 * v * x * ko_x * ko_y**2 * torch.cos(theta)
                )
                * torch.sin(theta)
                - 15
                * x
                * y
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                + 15
                * x
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                * ko_y
                + 15
                * y
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                * ko_x
                - 15
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                * ko_x
                * ko_y
                + (
                    -(r_sqr_err**2) * u_f * x
                    + r_sqr_err**2 * u_f * ko_x
                    + 3 * r_sqr_err * u_f * x * ko_y**2
                    + 6 * r_sqr_err * u_f * y * ko_x * ko_y
                    - 3 * r_sqr_err * u_f * ko_x * ko_y**2
                    + 3 * r_2 * u_f * x
                    - 3 * r_2 * u_f * ko_x
                    - 6 * r_3 * u_f * ko_y
                )
                * torch.sin(theta)
                + (
                    -(r_sqr_err**2) * u_f * y
                    + r_sqr_err**2 * u_f * ko_y
                    + 3 * r_sqr_err * r_6 * u_f * y
                    - 3 * r_sqr_err * r_6 * u_f * ko_y
                    + 6 * r_sqr_err * u_f * x * ko_x * ko_y
                    + 3 * r_1 * u_f * y
                    - 3 * r_1 * u_f * ko_y
                    - 6 * r_3 * u_f * ko_x
                )
                * torch.cos(theta)
            )
            / r_sqr_err ** (7 / 2),
            (
                v**2
                * (
                    -2 * r_sqr_err * y
                    + 2 * r_sqr_err * ko_y
                    + 6 * r_4
                    + 6 * r_6 * y
                    - 6 * r_6 * ko_y
                    - 6 * x**2 * ko_y
                    - 12 * x * y * ko_x
                    + 12 * x * ko_x * ko_y
                )
                * torch.cos(theta) ** 2
                + v**2
                * (
                    2 * r_sqr_err * y
                    - 2 * r_sqr_err * ko_y
                    - 6 * r_4
                    - 6 * r_6 * y
                    + 6 * r_6 * ko_y
                    + 6 * x**2 * ko_y
                    + 12 * x * y * ko_x
                    - 12 * x * ko_x * ko_y
                )
                * torch.sin(theta) ** 2
                + v
                * (
                    -omega * r_sqr_err**2
                    + omega * r_sqr_err * r_6
                    - 2 * omega * r_sqr_err * x * ko_x
                    + omega * r_1
                )
                * torch.cos(theta)
                + v
                * (
                    -omega * r_sqr_err * x * ko_y
                    - omega * r_sqr_err * y * ko_x
                    + omega * r_sqr_err * ko_x * ko_y
                    + omega * r_3
                    + 4 * r_sqr_err * v * x * torch.cos(theta)
                    - 4 * r_sqr_err * v * ko_x * torch.cos(theta)
                    + 6 * r_5 * v * torch.cos(theta)
                    - 18 * r_6 * v * x * torch.cos(theta)
                    - 6 * v * x**3 * torch.cos(theta)
                    + 18 * v * x**2 * ko_x * torch.cos(theta)
                    - 12 * v * x * y * ko_y * torch.cos(theta)
                    + 6 * v * x * ko_y**2 * torch.cos(theta)
                    - 6 * v * y**2 * ko_x * torch.cos(theta)
                    + 12 * v * y * ko_x * ko_y * torch.cos(theta)
                    + 6 * v * ko_x**3 * torch.cos(theta)
                    - 6 * v * ko_x * ko_y**2 * torch.cos(theta)
                )
                * torch.sin(theta)
                + (
                    -(r_sqr_err**2) * u_f
                    + r_sqr_err * r_6 * u_f
                    - 2 * r_sqr_err * u_f * x * ko_x
                    + r_1 * u_f
                )
                * torch.sin(theta)
                + (
                    r_sqr_err * u_f * x * ko_y
                    + r_sqr_err * u_f * y * ko_x
                    - r_sqr_err * u_f * ko_x * ko_y
                    - r_3 * u_f
                )
                * torch.cos(theta)
            )
            / r_sqr_err ** (5 / 2),
            (
                v
                * (
                    18 * r_6 * x * torch.cos(theta)
                    + 6 * x**3 * torch.cos(theta)
                    - 18 * x**2 * ko_x * torch.cos(theta)
                    - 6 * x * x_err**2 * torch.cos(theta)
                    - 6 * x * y_err**2 * torch.cos(theta)
                    + 6 * x_err**2 * ko_x * torch.cos(theta)
                    + 6 * y_err**2 * ko_x * torch.cos(theta)
                    - 6 * ko_x**3 * torch.cos(theta)
                )
                * torch.cos(theta)
                + v
                * (
                    4 * r_sqr_err * x * torch.sin(theta)
                    - 4 * r_sqr_err * y * torch.cos(theta)
                    - 4 * r_sqr_err * ko_x * torch.sin(theta)
                    + 4 * r_sqr_err * ko_y * torch.cos(theta)
                    + 12 * r_4 * torch.cos(theta)
                    + 6 * r_5 * torch.sin(theta)
                    + 12 * r_6 * y * torch.cos(theta)
                    - 12 * r_6 * ko_y * torch.cos(theta)
                    - 12 * x**2 * ko_y * torch.cos(theta)
                    - 6 * x * x_err**2 * torch.sin(theta)
                    - 24 * x * y * ko_x * torch.cos(theta)
                    - 12 * x * y * ko_y * torch.sin(theta)
                    - 6 * x * y_err**2 * torch.sin(theta)
                    + 24 * x * ko_x * ko_y * torch.cos(theta)
                    + 6 * x * ko_y**2 * torch.sin(theta)
                    + 6 * x_err**2 * ko_x * torch.sin(theta)
                    - 6 * y**2 * ko_x * torch.sin(theta)
                    + 12 * y * ko_x * ko_y * torch.sin(theta)
                    + 6 * y_err**2 * ko_x * torch.sin(theta)
                    - 6 * ko_x * ko_y**2 * torch.sin(theta)
                )
                * torch.sin(theta)
                + (
                    -omega * r_sqr_err**2
                    + omega * r_sqr_err * r_6
                    - 2 * omega * r_sqr_err * x * ko_x
                    + omega * r_1
                )
                * torch.sin(theta)
                + (
                    omega * r_sqr_err * x * ko_y
                    + omega * r_sqr_err * y * ko_x
                    - omega * r_sqr_err * ko_x * ko_y
                    - omega * r_3
                )
                * torch.cos(theta)
            )
            / r_sqr_err ** (5 / 2),
            v
            * (
                (-r_sqr_err + r_6 + x**2 - 2 * x * ko_x) * torch.sin(theta)
                + (-x * y + x * ko_y + y * ko_x - ko_x * ko_y) * torch.cos(theta)
            )
            / r_sqr_err ** (3 / 2),
        ],
    )
    coeff_hessian[3, :] = (
        [
            0,
            0,
            0,
            (
                k1 * k2 * r_sqr_err**3
                + 4 * k1 * k2 * r_sqr_err**2 * y * ko_y
                + k1 * k2 * r_sqr_err**2 * y_err**2
                - 2 * k1 * k2 * r_sqr_err**2 * ko_y**2
                - 2 * k1 * k2 * r_sqr_err * r_2
                + k2 * r_sqr_err**2 * x * ko_vel_x
                + 3 * k2 * r_sqr_err**2 * y * ko_vel_y
                - k2 * r_sqr_err**2 * ko_x * ko_vel_x
                - 3 * k2 * r_sqr_err**2 * ko_y * ko_vel_y
                - 3 * k2 * r_sqr_err * x * ko_y**2 * ko_vel_x
                - 6 * k2 * r_sqr_err * y * ko_x * ko_y * ko_vel_x
                - 9 * k2 * r_sqr_err * y * ko_y**2 * ko_vel_y
                + 3 * k2 * r_sqr_err * ko_x * ko_y**2 * ko_vel_x
                + 3 * k2 * r_sqr_err * ko_y**3 * ko_vel_y
                - 3 * k2 * r_2 * x * ko_vel_x
                - 3 * k2 * r_2 * y * ko_vel_y
                + 3 * k2 * r_2 * ko_x * ko_vel_x
                + 9 * k2 * r_2 * ko_y * ko_vel_y
                + 6 * k2 * r_3 * ko_y * ko_vel_x
                + r_sqr_err**2 * x * ko_vel_x
                + 3 * r_sqr_err**2 * y * ko_vel_y
                - r_sqr_err**2 * ko_x * ko_vel_x
                - 3 * r_sqr_err**2 * ko_y * ko_vel_y
                - r_sqr_err**2 * ko_vel_x**2
                - 3 * r_sqr_err**2 * ko_vel_y**2
                - 3 * r_sqr_err * x * ko_y**2 * ko_vel_x
                - 12 * r_sqr_err * x * ko_y * ko_vel_x * ko_vel_y
                - 6 * r_sqr_err * y * ko_x * ko_y * ko_vel_x
                - 12 * r_sqr_err * y * ko_x * ko_vel_x * ko_vel_y
                - 9 * r_sqr_err * y * ko_y**2 * ko_vel_y
                - 6 * r_sqr_err * y * ko_y * ko_vel_x**2
                - 30 * r_sqr_err * y * ko_y * ko_vel_y**2
                + 3
                * r_sqr_err
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                + 3 * r_sqr_err * ko_x * ko_y**2 * ko_vel_x
                + 12 * r_sqr_err * ko_x * ko_y * ko_vel_x * ko_vel_y
                + 3 * r_sqr_err * ko_y**3 * ko_vel_y
                + 3 * r_sqr_err * ko_y**2 * ko_vel_x**2
                + 15 * r_sqr_err * ko_y**2 * ko_vel_y**2
                - 3 * r_2 * x * ko_vel_x
                - 3 * r_2 * y * ko_vel_y
                + 3 * r_2 * ko_x * ko_vel_x
                + 9 * r_2 * ko_y * ko_vel_y
                + 3 * r_2 * ko_vel_x**2
                + 15 * r_2 * ko_vel_y**2
                + 6 * r_3 * ko_y * ko_vel_x
                + 12 * r_3 * ko_vel_x * ko_vel_y
                + v**2
                * (
                    -3 * r_sqr_err * x_err**2
                    - 6 * r_sqr_err * y * ko_y
                    - 3 * r_sqr_err * y_err**2
                    + 3 * r_sqr_err * ko_y**2
                    + 3 * r_2
                    + 15 * x_err**2 * y**2
                    - 30 * x_err**2 * y * ko_y
                    + 15 * x_err**2 * ko_y**2
                    - 15 * y**4
                    + 60 * y**3 * ko_y
                    + 15 * y**2 * y_err**2
                    - 90 * y**2 * ko_y**2
                    - 30 * y * y_err**2 * ko_y
                    + 60 * y * ko_y**3
                    + 15 * y_err**2 * ko_y**2
                    - 15 * ko_y**4
                )
                * torch.sin(theta) ** 2
                + v**2
                * (
                    2 * r_sqr_err**2
                    + 3 * r_sqr_err * r_6
                    - 6 * r_sqr_err * x * ko_x
                    - 3 * r_sqr_err * x_err**2
                    + 24 * r_sqr_err * y * ko_y
                    - 3 * r_sqr_err * y_err**2
                    - 12 * r_sqr_err * ko_y**2
                    + 3 * r_1
                    - 12 * r_2
                    - 15 * r_4 * y
                    + 30 * r_4 * ko_y
                    + 30 * r_5 * ko_x
                    - 15 * r_6 * y**2
                    + 30 * r_6 * y * ko_y
                    - 15 * r_6 * ko_y**2
                    - 15 * x**2 * ko_y**2
                    - 60 * x * y * ko_x * ko_y
                    + 30 * x * ko_x * ko_y**2
                    + 15 * x_err**2 * y**2
                    - 30 * x_err**2 * y * ko_y
                    + 15 * x_err**2 * ko_y**2
                    + 15 * y**2 * y_err**2
                    - 30 * y * y_err**2 * ko_y
                    + 15 * y_err**2 * ko_y**2
                )
                * torch.cos(theta) ** 2
                + v
                * (
                    -3 * omega * r_sqr_err**2 * y
                    + 3 * omega * r_sqr_err**2 * ko_y
                    + 9 * omega * r_sqr_err * y * ko_y**2
                    - 3 * omega * r_sqr_err * ko_y**3
                    + 3 * omega * r_2 * y
                    - 9 * omega * r_2 * ko_y
                )
                * torch.cos(theta)
                + v
                * (
                    omega * r_sqr_err**2 * x
                    - omega * r_sqr_err**2 * ko_x
                    - 3 * omega * r_sqr_err * x * ko_y**2
                    - 6 * omega * r_sqr_err * y * ko_x * ko_y
                    + 3 * omega * r_sqr_err * ko_x * ko_y**2
                    - 3 * omega * r_2 * x
                    + 3 * omega * r_2 * ko_x
                    + 6 * omega * r_3 * ko_y
                    - 18 * r_sqr_err * v * x * ko_y * torch.cos(theta)
                    - 18 * r_sqr_err * v * y * ko_x * torch.cos(theta)
                    + 18 * r_sqr_err * v * ko_x * ko_y * torch.cos(theta)
                    + 18 * r_3 * v * torch.cos(theta)
                    - 30 * r_5 * v * y * torch.cos(theta)
                    + 90 * r_5 * v * ko_y * torch.cos(theta)
                    - 90 * v * x * y * ko_y**2 * torch.cos(theta)
                    + 30 * v * x * ko_y**3 * torch.cos(theta)
                    + 30 * v * y**3 * ko_x * torch.cos(theta)
                    - 90 * v * y**2 * ko_x * ko_y * torch.cos(theta)
                    + 90 * v * y * ko_x * ko_y**2 * torch.cos(theta)
                    - 30 * v * ko_x * ko_y**3 * torch.cos(theta)
                )
                * torch.sin(theta)
                - 15
                * y**2
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                + 30
                * y
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                * ko_y
                - 15
                * (-x * ko_vel_x - y * ko_vel_y + ko_x * ko_vel_x + ko_y * ko_vel_y)
                ** 2
                * ko_y**2
                + (
                    -3 * r_sqr_err**2 * u_f * y
                    + 3 * r_sqr_err**2 * u_f * ko_y
                    + 9 * r_sqr_err * u_f * y * ko_y**2
                    - 3 * r_sqr_err * u_f * ko_y**3
                    + 3 * r_2 * u_f * y
                    - 9 * r_2 * u_f * ko_y
                )
                * torch.sin(theta)
                + (
                    -(r_sqr_err**2) * u_f * x
                    + r_sqr_err**2 * u_f * ko_x
                    + 3 * r_sqr_err * u_f * x * ko_y**2
                    + 6 * r_sqr_err * u_f * y * ko_x * ko_y
                    - 3 * r_sqr_err * u_f * ko_x * ko_y**2
                    + 3 * r_2 * u_f * x
                    - 3 * r_2 * u_f * ko_x
                    - 6 * r_3 * u_f * ko_y
                )
                * torch.cos(theta)
            )
            / r_sqr_err ** (7 / 2),
            (
                v**2
                * (
                    -2 * r_sqr_err * x
                    + 2 * r_sqr_err * ko_x
                    + 6 * r_5
                    - 12 * x * y * ko_y
                    + 6 * x * ko_y**2
                    - 6 * y**2 * ko_x
                    + 12 * y * ko_x * ko_y
                    - 6 * ko_x * ko_y**2
                )
                * torch.cos(theta) ** 2
                + v**2
                * (
                    2 * r_sqr_err * x
                    - 2 * r_sqr_err * ko_x
                    - 6 * r_5
                    + 12 * x * y * ko_y
                    - 6 * x * ko_y**2
                    + 6 * y**2 * ko_x
                    - 12 * y * ko_x * ko_y
                    + 6 * ko_x * ko_y**2
                )
                * torch.sin(theta) ** 2
                + v
                * (
                    -omega * r_sqr_err * x * ko_y
                    - omega * r_sqr_err * y * ko_x
                    + omega * r_sqr_err * ko_x * ko_y
                    + omega * r_3
                )
                * torch.cos(theta)
                + v
                * (
                    -omega * r_sqr_err**2
                    - 2 * omega * r_sqr_err * y * ko_y
                    + omega * r_sqr_err * ko_y**2
                    + omega * r_2
                    - 4 * r_sqr_err * v * y * torch.cos(theta)
                    + 4 * r_sqr_err * v * ko_y * torch.cos(theta)
                    - 6 * r_4 * v * torch.cos(theta)
                    - 6 * r_6 * v * y * torch.cos(theta)
                    + 6 * r_6 * v * ko_y * torch.cos(theta)
                    + 6 * v * x**2 * ko_y * torch.cos(theta)
                    + 12 * v * x * y * ko_x * torch.cos(theta)
                    - 12 * v * x * ko_x * ko_y * torch.cos(theta)
                    + 6 * v * y**3 * torch.cos(theta)
                    - 18 * v * y**2 * ko_y * torch.cos(theta)
                    + 18 * v * y * ko_y**2 * torch.cos(theta)
                    - 6 * v * ko_y**3 * torch.cos(theta)
                )
                * torch.sin(theta)
                + (
                    r_sqr_err**2 * u_f
                    + 2 * r_sqr_err * u_f * y * ko_y
                    - r_sqr_err * u_f * ko_y**2
                    - r_2 * u_f
                )
                * torch.cos(theta)
                + (
                    -r_sqr_err * u_f * x * ko_y
                    - r_sqr_err * u_f * y * ko_x
                    + r_sqr_err * u_f * ko_x * ko_y
                    + r_3 * u_f
                )
                * torch.sin(theta)
            )
            / r_sqr_err ** (5 / 2),
            (
                v
                * (
                    4 * r_sqr_err * y * torch.cos(theta)
                    - 4 * r_sqr_err * ko_y * torch.cos(theta)
                    + 6 * r_4 * torch.cos(theta)
                    + 6 * r_6 * y * torch.cos(theta)
                    - 6 * r_6 * ko_y * torch.cos(theta)
                    - 6 * x**2 * ko_y * torch.cos(theta)
                    - 12 * x * y * ko_x * torch.cos(theta)
                    + 12 * x * ko_x * ko_y * torch.cos(theta)
                    - 6 * x_err**2 * y * torch.cos(theta)
                    + 6 * x_err**2 * ko_y * torch.cos(theta)
                    - 6 * y * y_err**2 * torch.cos(theta)
                    + 6 * y_err**2 * ko_y * torch.cos(theta)
                )
                * torch.cos(theta)
                + v
                * (
                    -4 * r_sqr_err * x * torch.cos(theta)
                    + 4 * r_sqr_err * ko_x * torch.cos(theta)
                    + 12 * r_5 * torch.cos(theta)
                    - 24 * x * y * ko_y * torch.cos(theta)
                    + 12 * x * ko_y**2 * torch.cos(theta)
                    - 6 * x_err**2 * y * torch.sin(theta)
                    + 6 * x_err**2 * ko_y * torch.sin(theta)
                    + 6 * y**3 * torch.sin(theta)
                    - 12 * y**2 * ko_x * torch.cos(theta)
                    - 18 * y**2 * ko_y * torch.sin(theta)
                    - 6 * y * y_err**2 * torch.sin(theta)
                    + 24 * y * ko_x * ko_y * torch.cos(theta)
                    + 18 * y * ko_y**2 * torch.sin(theta)
                    + 6 * y_err**2 * ko_y * torch.sin(theta)
                    - 12 * ko_x * ko_y**2 * torch.cos(theta)
                    - 6 * ko_y**3 * torch.sin(theta)
                )
                * torch.sin(theta)
                + (
                    omega * r_sqr_err**2
                    + 2 * omega * r_sqr_err * y * ko_y
                    - omega * r_sqr_err * ko_y**2
                    - omega * r_2
                )
                * torch.cos(theta)
                + (
                    -omega * r_sqr_err * x * ko_y
                    - omega * r_sqr_err * y * ko_x
                    + omega * r_sqr_err * ko_x * ko_y
                    + omega * r_3
                )
                * torch.sin(theta)
            )
            / r_sqr_err ** (5 / 2),
            v
            * (
                (r_sqr_err - y**2 + 2 * y * ko_y - ko_y**2) * torch.cos(theta)
                + (x * y - x * ko_y - y * ko_x + ko_x * ko_y) * torch.sin(theta)
            )
            / r_sqr_err ** (3 / 2),
        ],
    )
    coeff_hessian[4, :] = [
        0,
        0,
        0,
        0,
        (
            v**2
            * (
                -2 * r_6
                - 2 * x**2
                + 4 * x * ko_x
                + 2 * y**2
                - 4 * y * ko_y
                + 2 * ko_y**2
            )
            * torch.sin(theta) ** 2
            + v**2
            * (
                2 * r_6
                + 2 * x**2
                - 4 * x * ko_x
                - 2 * y**2
                + 4 * y * ko_y
                - 2 * ko_y**2
            )
            * torch.cos(theta) ** 2
            + v * (-omega * r_sqr_err * y + omega * r_sqr_err * ko_y) * torch.cos(theta)
            + v
            * (
                omega * r_sqr_err * x
                - omega * r_sqr_err * ko_x
                + 8 * v * x * y * torch.cos(theta)
                - 8 * v * x * ko_y * torch.cos(theta)
                - 8 * v * y * ko_x * torch.cos(theta)
                + 8 * v * ko_x * ko_y * torch.cos(theta)
            )
            * torch.sin(theta)
            + (-r_sqr_err * u_f * x + r_sqr_err * u_f * ko_x) * torch.cos(theta)
            + (-r_sqr_err * u_f * y + r_sqr_err * u_f * ko_y) * torch.sin(theta)
        )
        / r_sqr_err ** (3 / 2),
        (
            v
            * (
                -4 * x * y * torch.cos(theta)
                + 4 * x * ko_y * torch.cos(theta)
                + 4 * y * ko_x * torch.cos(theta)
                - 4 * ko_x * ko_y * torch.cos(theta)
            )
            * torch.cos(theta)
            + v
            * (
                4 * r_6 * torch.cos(theta)
                + 4 * x**2 * torch.cos(theta)
                + 4 * x * y * torch.sin(theta)
                - 8 * x * ko_x * torch.cos(theta)
                - 4 * x * ko_y * torch.sin(theta)
                - 4 * y**2 * torch.cos(theta)
                - 4 * y * ko_x * torch.sin(theta)
                + 8 * y * ko_y * torch.cos(theta)
                + 4 * ko_x * ko_y * torch.sin(theta)
                - 4 * ko_y**2 * torch.cos(theta)
            )
            * torch.sin(theta)
            + (-omega * r_sqr_err * x + omega * r_sqr_err * ko_x) * torch.cos(theta)
            + (-omega * r_sqr_err * y + omega * r_sqr_err * ko_y) * torch.sin(theta)
        )
        / r_sqr_err ** (3 / 2),
        v
        * ((-x + ko_x) * torch.cos(theta) + (-y + ko_y) * torch.sin(theta))
        / torch.sqrt(r_sqr_err),
    ]
    coeff_hessian[5, :] = [
        0,
        0,
        0,
        0,
        0,
        (
            (
                -4 * x * y * torch.cos(theta)
                + 4 * x * ko_y * torch.cos(theta)
                + 4 * y * ko_x * torch.cos(theta)
                - 4 * ko_x * ko_y * torch.cos(theta)
            )
            * torch.sin(theta)
            + (-2 * r_6 - 2 * x**2 + 4 * x * ko_x + 2 * x_err**2 + 2 * y_err**2)
            * torch.cos(theta) ** 2
            + (2 * x_err**2 - 2 * y**2 + 4 * y * ko_y + 2 * y_err**2 - 2 * ko_y**2)
            * torch.sin(theta) ** 2
        )
        / r_sqr_err ** (3 / 2),
        ((-x + ko_x) * torch.sin(theta) + (y - ko_y) * torch.cos(theta))
        / torch.sqrt(r_sqr_err),
    ]
    coeff_hessian[6, :] = [0, 0, 0, 0, 0, 0, 0]

    # restores the symmetric lower half
    coeff_hessian += coeff_hessian.T - torch.diag(coeff_hessian)

    # computes the correction utilizing the connection
    conn_coeffs = cfg.conn(u)
    return coeff_hessian - torch.tensordot(u, conn_coeffs, ([0], [0]))


def gcbf_f(u: torch.Tensor, mfld_cfg: MfldCfg, u_nom: torch.Tensor):
    # minimization of correction between nominal control input and safe input
    return 0.5 * dist_squared_map(u, u_nom, mfld_cfg)  # >= 0


class GCBF_Cost(MfldFunc):
    def __new__(cls):
        return 0.5 * RiemannSquaredDist()


class GCBF_Constraint(MfldFunc):
    # NOTE: the negatives are to translate this from the usual >= notation of
    # CBFs used in our derivation to a g <= 0 as is commonly employd within
    # solvers and the manifold solver used in this library

    def value(self, u, _cfg, _u_nom, p, ko, k1, k2):
        return -cbf_ko(p, u, ko, k1, k2)

    def diff(self, u, _cfg, _u_nom, p, ko, k1, k2):
        return -cbf_ko_diff(p, u, ko, k1, k2)

    def hess(self, u, cfg, _u_nom, p, ko, k1, k2):
        return -cbf_ko_hess(p, u, ko, k1, k2, cfg)
