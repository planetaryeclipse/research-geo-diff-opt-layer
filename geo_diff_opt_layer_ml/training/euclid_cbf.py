import torch


def batched_cbf_ko_coeffs(
    p: torch.Tensor,  # batched coordinates
    ko: torch.Tensor,  # keep-out regions
    k_1: torch.Tensor,  # constant positive parameter for cbf
    k_2: torch.Tensor,  # constant positive parameter for cbf
):
    # note that this formulation splits the cbf formulation into a leading
    # coefficient for the single input u_f (u_t does not appear in the CBF) and
    # a linear term that then forms a linear constraint in implementation

    # splits the state and keep-out regions for convenience in the following
    # operations but we have to take the column vector given the batched forms
    (x, y, theta, v, omega) = (p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4])
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
        ko[:, 0],
        ko[:, 1],
        ko[:, 2],
        ko[:, 3],
        ko[:, 4],
        ko[:, 5],
        ko[:, 6],
        ko[:, 7],
        ko[:, 8],
    )

    x_err = ko_x - x
    y_err = ko_y - y
    r_sqr_err = x_err**2 + y_err**2

    r_1 = r_sqr_err ** (7.0 / 2) * x_err
    r_2 = r_sqr_err ** (7.0 / 2) * y_err
    r_3 = r_sqr_err ** (5.0 / 2) * x_err**2
    r_4 = r_sqr_err ** (5.0 / 2) * y_err**2
    r_5 = r_sqr_err ** (5.0 / 2) * x_err * y_err

    # the coefficient of u_f
    a_coeffs = torch.zeros((p.shape[0], 1, 2))
    a_coeffs[:, 0, 0] = (r_1 * torch.cos(theta) + r_2 * torch.sin(theta)) / r_sqr_err**4

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
        - 2 * r_sqr_err ** (7.0 / 2) * ko_vel_x**2
        - 2 * r_sqr_err ** (7.0 / 2) * ko_vel_y**2
        + 2 * r_sqr_err**4 * ko_accel_radius
    ) / (2 * r_sqr_err**4)

    return a_coeffs, b_term.reshape((b_term.shape[0], 1))
