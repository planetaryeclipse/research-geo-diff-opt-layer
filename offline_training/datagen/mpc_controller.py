import numpy as np

from dataclasses import dataclass, field

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from typing import Any, Callable, Tuple

# @dataclass
# class Trajectory:
#     # utility class to define and get values for a trajectory when computing
#     # the cost due to the tracking error (to be passed along with the params)

#     dt: float
#     t: np.ndarray
#     x_traj: np.ndarray
#     y_traj: np.ndarray
#     dx_traj: np.ndarray
#     dy_traj: np.ndarray
#     ddx_traj: np.ndarray
#     ddy_traj: np.ndarray

#     def __call__(self, t: float) -> Tuple[float, float, float, float, float, float]:
#         return (
#             np.interp(t, self.t, self.x_traj),
#             np.interp(t, self.t, self.y_traj),
#             np.interp(t, self.t, self.dx_traj),
#             np.interp(t, self.t, self.dy_traj),
#             np.interp(t, self.t, self.ddx_traj),
#             np.interp(t, self.t, self.ddy_traj),
#        )




def _mpc_cost(
    t: float,
    y: np.ndarray,
    u_steps: np.ndarray,
    dt: float,
    params: Any,
    model_step: Callable[[float, float, np.ndarray, np.ndarray, Any], np.ndarray],
    state_cost: Callable[[float, np.ndarray, np.ndarray, int, Any], float],
) -> float:
    # computes the cost for the current set of control inputs u defined over
    # the upcoming finite horizon (rows of u_steps)
    # NOTE: optimize the costs in future steps, do not consider this step

    num_steps = u_steps.shape[0]

    total_cost = 0
    y_upd = y

    for i in range(num_steps):
        u = u_steps[i, :]
        t_upd = t + dt * i

        y_upd = model_step(t_upd, dt, y_upd, u, params)
        total_cost += state_cost(t_upd, y_upd, u, i, params)

    return total_cost


def gen_mpc_controls(
    t: float,
    y: np.ndarray,
    steps: int,
    dt: float,
    u_guess: np.ndarray,
    params: Any,
    model_step: Callable[[float, float, np.ndarray, np.ndarray, Any], np.ndarray],
    state_cost: Callable[[float, np.ndarray, np.ndarray, int, Any], float],
):
    # must find inputs over finite time horizon that minimize the total cost

    # repeats the guess then flattens to be able to use as variable for minimize
    u_steps_guess = np.repeat(u_guess.reshape((1, len(u_guess))), steps, axis=0).flatten()

    result = minimize(
        lambda u_steps: _mpc_cost(
            t,
            y,
            u_steps.reshape((steps, len(u_guess))),  # each step as row
            dt,
            params,
            model_step,
            state_cost,
        ),
        u_steps_guess,
    )

    # NOTE: useful for debugging but was interfering with tqdm readout
    # if not result.success:
    #     print("Failed to fully converge at solution, is suboptimal")

    result_u_steps = result.x.reshape((steps, len(u_guess)))
    return (result_u_steps[0, :], result_u_steps)
