import torch
from diff_mfld_optim.mfld_util import MfldCfg, dist_squared_map

# NOTE: these are the methods necessary to implement the keep-out region for
# the geometric formulation of the training


def cbf_ko(
    p: torch.Tensor,  # current unicycle state
    u: torch.Tensor,  # current input
    p_ko: torch.Tensor,  # ko centre position
    dp_ko: torch.Tensor,  # ko centre velocity
    ddp_ko: torch.Tensor,  # ko centre acceleration
    r_ko: torch.Tensor,  # ko radius
    dr_ko: torch.Tensor,  # ko radius velocity
    ddr_ko: torch.Tensor,  # ko radius acceleration
    k_1: torch.Tensor,
    k_2: torch.Tensor,
):
    # evaluates the ghocbf for our system using pytorch (so that it is
    # can be differentiated using autograd to be used in a constrained
    # optimization solver)

    # for clarity expand the above elements (function signature kept in a more
    # convenient grouping rather than each individual element)
    x, y, theta, v, omega = p[0], p[1], p[2], p[3], p[4]

    x_ko, y_ko = p_ko[0], p_ko[1]
    dx_ko, dy_ko = dp_ko[0], dp_ko[1]
    ddx_ko, ddy_ko = ddp_ko[0], ddp_ko[1]

    u_f, _u_t = u[0], u[1]

    # for details refer to `dyn_ext_unicycle_ko_cbf_derivations.ipynb`
    x_err = x_ko - x
    y_err = y_ko - y
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
        + 2 * k_1 * k_2 * r_sqr_err**4 * r_ko
        - 2 * k_2 * r_1 * dx_ko
        - 2 * k_2 * r_2 * dy_ko
        + 2 * k_2 * r_sqr_err**4 * dr_ko
        - 2 * r_1 * torch.sin(theta) * v * omega
        - 2 * r_1 * ddx_ko
        + 2 * r_2 * torch.cos(theta) * v * omega
        - 2 * r_2 * ddy_ko
        + r_3 * torch.cos(2 * theta) * v**2
        + r_3 * v**3
        + 2 * r_3 * ddx_ko**2
        - r_4 * torch.cos(2 * theta) * v**2
        + r_4 * v**2
        + 2 * r_4 * (ddy_ko) ** 2
        + 2 * r_5 * torch.sin(2 * theta) * v**2
        + 4 * r_5 * dx_ko * dy_ko
        - 2 * r_sqr_err ** (7.0 / 2) * v**2
        - 2 * r_sqr_err ** (7.0 / 2) * (dx_ko) ** 2
        - 2 * r_sqr_err ** (7.0 / 2) * (dy_ko) ** 2
        + 2 * r_sqr_err**4 * ddr_ko
    ) / (2 * r_sqr_err**4)

    return a_coeff + b_term


def gcbf_f(u: torch.Tensor, mfld_cfg: MfldCfg, u_nom: torch.Tensor):
    # minimization of correction between nominal control input and safe input
    return 0.5 * dist_squared_map(u, u_nom, mfld_cfg)
