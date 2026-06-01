from dataclasses import dataclass
from typing import Callable

import numpy as np

from scipy.integrate import solve_ivp

DYN_EXT_UNICYCLE_STATE_LEN = 5
DYN_EXT_UNICYCLE_CONTROLS_LEN = 2


@dataclass
class DynExtUnicycleMCPParams:
    # defines the necessary parameters used to compute the cost due to the
    # state and control input error when used in the mpc controller

    dist_cost: float
    ang_cost: float
    u_cost: np.ndarray
    fut_pred_factor: Callable[[int], float]
    traj: Trajectory  # must be defined before passing this


def dyn_ext_unicycle_model_step(
    t: float, dt: float, y: np.ndarray, u: np.ndarray, _params: DynExtUnicycleMCPParams
) -> np.ndarray:
    # defines the behavior of the dynamic unicycle
    u_f, u_t = u

    def model_f(t, y):
        _, _, theta, v, omega = y
        return np.array([v * np.cos(theta), v * np.sin(theta), omega, u_f, u_t])

    result = solve_ivp(model_f, [0, dt], y)
    return result.y[:, -1]


def dyn_ext_unicycle_cost(t: float, y: np.ndarray, u: np.ndarray, k: int, params: DynExtUnicycleMCPParams) -> float:
    # defines the current cost considering both the state and inputs for the
    # dynamic unicycle to be used in the mpc controller

    x, y, theta, _v, _omega = y
    x_traj, y_traj, dx_traj, dy_traj, _, _ = params.traj(t)

    x_err = x_traj - x
    y_err = y_traj - y

    traj_theta = np.atan2(dy_traj, dx_traj)
    theta_err_sqr = np.minimum(
        (traj_theta - theta) ** 2,
        (np.mod(theta + 2 * np.pi, 2 * np.pi) - np.mod(traj_theta + 2 * np.pi, 2 * np.pi)) ** 2,
    )

    state_cost = 0.5 * params.dist_cost * (x_err**2 + y_err**2) + 0.5 * params.ang_cost * theta_err_sqr

    input_cost = u.T @ params.u_cost @ u

    return params.fut_pred_factor(k) * state_cost + input_cost
