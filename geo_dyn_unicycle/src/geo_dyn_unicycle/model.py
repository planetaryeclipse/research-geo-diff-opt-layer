import numpy as np

from scipy.integrate import solve_ivp

DYN_EXT_UNICYCLE_STATE_LEN = 5
DYN_EXT_UNICYCLE_CONTROLS_LEN = 2


def dyn_ext_unicycle_model_step(
    t: float, dt: float, y: np.ndarray, u: np.ndarray, **kargs
) -> np.ndarray:
    # defines the behavior of the dynamic unicycle
    u_f, u_t = u

    def model_f(t, y):
        _, _, theta, v, omega = y
        return np.array([v * np.cos(theta), v * np.sin(theta), omega, u_f, u_t])

    result = solve_ivp(model_f, [0, dt], y)
    return result.y[:, -1]
