import sys
import numpy as np

from dataclasses import dataclass
from pathlib import Path

from typing import Callable

sys.path.append(str(Path(__file__).parent))


from archive.mpc_controller.mpc_controller import gen_mpc_controls

from offline_training.datagen.util.episode_gen import Episode
from util import EPISODES_TRAIN_DIR, EPISODES_VALID_DIR, MPC_TRAIN_DIR, MPC_VALID_DIR
from dyn_unicycle import (
    DYN_EXT_UNICYCLE_STATE_LEN,
    DYN_EXT_UNICYCLE_CONTROLS_LEN,
    DynExtUnicycleMCPParams,
    dyn_ext_unicycle_model_step,
    dyn_ext_unicycle_cost,
)

from episode_gen import DynUnicycleEpisode, DynUnicycleTrajectory


@dataclass
class MPCEpisode:
    episode: DynUnicycleEpisode
    states: np.ndarray
    controls: np.ndarray


def gen_controls_for_episode(
    episode: DynUnicycleEpisode,
    dist_err_cost: float,
    ang_err_cost: float,
    controls_cost: np.ndarray,
    fut_pred_cost_factor: Callable[[int], float],
    num_mpc_steps: int,
    mpc_dt: float,
) -> MPCEpisode:
    if episode.trajectory.x[0].shape[1] != 2:
        raise ValueError("provided episode is not (x,y)")

    raise NotImplementedError("mpc not fully re-implemented yet")

    # states are (x, y, theta, v, omega), controls are (u_f, u_t)
    # episode only contains the starting position and orientation so the rest are assumed zero
    curr_state = np.array([*episode.start, 0.0, 0.0])
    curr_controls = np.zeros((DYN_EXT_UNICYCLE_CONTROLS_LEN))

    params = DynExtUnicycleMCPParams(
        dist_err_cost,
        ang_err_cost,
        controls_cost,
        fut_pred_cost_factor,
        episode.trajectory,
    )

    for i in range(len(episode.trajectory.t)):
        t_curr = episode.trajectory.t[i]
        traj_curr = episode.trajectory.x[i, :]

        u_optimal, _ = gen_mpc_controls(
            t_curr,
            curr_state,
            num_mpc_steps,
            mpc_dt,
            curr_controls,
            params,
            dyn_ext_unicycle_model_step,
            dyn_ext_unicycle_cost,
        )

        pass

    pass


# def gen_controls_for_episode(
#     episode,
#     dist_cost,
#     ang_cost,
#     u_cost,
#     future_pred_cost_factor,
#     num_mpc_steps,
# ):
#     # generates the state and control history for the episode

#     p_hist = []  # state history
#     u_hist = []  # control history

#     traj_err_hist = []  # position tracking error

#     # really only care about the randomized position as given the distribution
#     # it will have angles distributed with respect to the ideal trajectory
#     # anyways so this should be fine
#     # NOTE: also start at zero velocity or angular velocity
#     p_curr = np.array([*episode["start_pos"], 0.0, 0.0, 0.0])
#     u_curr = np.zeros(2)

#     # print(f"Start pos: {p_curr}")

#     dt = episode["sample_time"]

#     # setup the trajectory and parameters
#     t = episode["t_traj"]

#     traj = Trajectory(
#         dt,
#         t,
#         episode["x_traj"],
#         episode["y_traj"],
#         episode["dx_traj"],
#         episode["dy_traj"],
#         episode["ddx_traj"],
#         episode["ddy_traj"],
#     )

#     params = DynExtUnicycleMCPParams(
#         dist_cost, ang_cost, u_cost, future_pred_cost_factor, traj
#     )

#     for i in range(len(t)):
#         t_curr = t[i]

#         # print(f"t_curr: {t_curr}")

#         u_optimal, _ = gen_mpc_controls(
#             t_curr,
#             p_curr,
#             num_mpc_steps,
#             mpc_dt,
#             u_curr,
#             params,
#             dyn_ext_unicycle_model_step,
#             dyn_ext_unicycle_cost,
#         )

#         p_hist.append(p_curr)
#         u_hist.append(u_optimal)

#         # NOTE: technically this trajectory error can be computed afterwards but
#         # for convenience it will be computed here and is immediately available
#         x_traj_curr, y_traj_curr, _, _, _, _ = traj(t_curr)
#         traj_err = np.array([x_traj_curr - p_curr[0], y_traj_curr - p_curr[1]])
#         traj_err_hist.append(traj_err)

#         # updates the current position afterwards so it considers the initial
#         # condition when generating the frist control input
#         p_curr = dyn_ext_unicycle_model_step(t_curr, dt, p_curr, u_optimal, params)

#         # for testing
#         # if i > 3000:
#         #     break

#     return np.array(p_hist), np.array(u_hist), np.array(traj_err_hist)
