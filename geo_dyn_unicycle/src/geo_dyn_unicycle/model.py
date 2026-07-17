import numpy as np

from scipy.integrate import solve_ivp

DYN_EXT_UNICYCLE_STATE_LEN = 5
DYN_EXT_UNICYCLE_CONTROLS_LEN = 2

ROBOMASTER_MAX_SPEED = 2.8  # m/s
ROBOMASTER_MAX_ANG_RATE = 10.472  # rad/s


def dyn_ext_unicycle_model_step(
    t: float,
    dt: float,
    y: np.ndarray,
    u: np.ndarray,
    max_v: float,
    max_omega: float,
    **kargs
) -> np.ndarray:
    # defines the behavior of the dynamic unicycle
    u_f, u_t = u

    # print(f"t: {t}")
    # print(f"dt: {dt}")
    # print(f"y: {y}")
    # print(f"u: {u}")

    def model_f(t, y):
        _, _, theta, v, omega = y

        # enforces maximum velocities
        u_f_eff = u_f
        if v >= max_v and u_f > 0.0:
            u_f_eff = 0.0
        elif v <= -max_v and u_f < 0.0:
            u_f_eff = 0.0

        u_t_eff = u_t
        if omega >= max_omega and u_t > 0.0:
            u_t_eff = 0.0
        elif omega <= -max_omega and u_t < 0.0:
            u_t_eff = 0.0

        return np.array([v * np.cos(theta), v * np.sin(theta), omega, u_f_eff, u_t_eff])

    result = solve_ivp(model_f, [0, dt], y)
    return result.y[:, -1]
