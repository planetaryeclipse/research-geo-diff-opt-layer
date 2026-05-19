import torch
from diff_mfld_optim.mfld_util import MfldCfg, dist_squared_map


from diff_mfld_optim.geometry.funcs import (
    MfldFunc,
    RiemannSquaredDist,
)

# as these notebooks implement the GHOCBF-QP for the non-product formulation
# in which optimization occurs on the IST then the gradient and covariant
# hessian therefore differ


def cbf_ko(
    p: torch.Tensor,  # current unicycle state
    u: torch.Tensor,  # current input
    ko: torch.Tensor,  # keep-out coniguration
    k1: torch.Tensor,  # cbf param
    k2: torch.Tensor,  # cbf param
):
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
):
    (x, y, theta) = (p[0], p[1], p[2])
    (
        ko_x,
        ko_y,
    ) = (
        ko[0],
        ko[1],
    )
    u_f = u[0]

    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err ** (7.0 / 2) * x_err
    r_2 = r_sqr_err ** (7.0 / 2) * y_err

    # for clarity we split the full condition into a linear system by finding a
    # leading coefficient for the u_f term and a constant
    a_coeff = -(+2 * r_1 * torch.cos(theta) + 2 * r_2 * torch.sin(theta)) / (
        2 * r_sqr_err**4
    )

    return torch.tensor([a_coeff, 0.0])


def cbf_ko_hess(u: torch.Tensor, cfg: MfldCfg):
    conn_coeffs = cfg.conn(u)
    return -torch.tensordot(u, conn_coeffs, ([0], [0]))


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

    def value(self, u, cfg, _u_nom, p, ko, k1, k2):
        return -cbf_ko(p, u, ko, k1, k2)

    def diff(self, u, cfg, _u_nom, p, ko, _k1, _k2):
        return -cbf_ko_diff(p, u, ko)

    def hess(self, u, cfg, _u_nom, _p, _ko, _k1, _k2):
        return -cbf_ko_hess(u, cfg)
