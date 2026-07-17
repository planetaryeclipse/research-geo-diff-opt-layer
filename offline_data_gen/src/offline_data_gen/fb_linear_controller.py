import numpy as np

from dataclasses import dataclass

from offline_data_gen.episode_gen import DynUnicycleEpisode
from offline_data_gen.util import Serializable
from geo_dyn_unicycle.model import (
    DYN_EXT_UNICYCLE_CONTROLS_LEN,
    DYN_EXT_UNICYCLE_STATE_LEN,
    dyn_ext_unicycle_model_step,
)


@dataclass
class SimulationResult(Serializable):
    states: np.ndarray
    controls: np.ndarray


def simulate_under_fb_linear_control(
    episode: DynUnicycleEpisode,
    kp_gains: np.ndarray,
    kd_gains: np.ndarray,
    offset: float,
    min_controls: np.ndarray,
    max_controls: np.ndarray,
    max_v: float,
    max_omega: float,
    show_debug: bool = False,
) -> SimulationResult:
    if episode.trajectory.x.shape[1] != 2:
        raise ValueError("provided episode is not (x,y)")

    # states are (x, y, theta, v, omega), controls are (u_f, u_t)
    curr_t = episode.trajectory.t[0]
    curr_state = np.array(
        [
            *episode.start,  # x, y, theta
            0.0,
            0.0,  # starts without velocities
        ]
    )
    curr_controls = None

    num_timesteps = len(episode.trajectory.t)
    dt = episode.trajectory.t[1] - episode.trajectory.t[0]

    state_hist = np.zeros((num_timesteps, DYN_EXT_UNICYCLE_STATE_LEN))
    controls_hist = np.zeros((num_timesteps, DYN_EXT_UNICYCLE_CONTROLS_LEN))

    for i in range(num_timesteps):
        if show_debug:
            print(f"step: {i}/{num_timesteps}")

        # gets the current state and trajectory state
        curr_t = episode.trajectory.t[i]

        curr_theta, curr_lin_vel, curr_ang_vel = curr_state[2:]
        curr_pos = curr_state[:2]

        traj_pos = episode.trajectory.x[i, :]
        traj_vel = episode.trajectory.dx[i, :]
        traj_accel = episode.trajectory.ddx[i, :]

        a_mat = np.array(
            [
                [np.cos(curr_theta), -offset * np.sin(curr_theta)],
                [np.sin(curr_theta), offset * np.cos(curr_theta)],
            ]
        )
        point_pos = curr_pos + offset * np.array(
            [np.cos(curr_theta), np.sin(curr_theta)]
        )
        point_vel = a_mat @ np.array([curr_lin_vel, curr_ang_vel])
        nl_dyn_term = np.array(
            [
                -curr_lin_vel * np.sin(curr_theta) * curr_ang_vel
                - offset * np.cos(curr_theta) * curr_ang_vel**2,
                curr_lin_vel * np.cos(curr_theta) * curr_ang_vel
                - offset * np.sin(curr_theta) * curr_ang_vel**2,
            ]
        )

        err_vel = traj_vel - point_vel
        err_pos = traj_pos - point_pos

        if show_debug:
            print(f"err_pos: {err_pos}")

        curr_controls_unsat = np.linalg.inv(a_mat) @ (
            -nl_dyn_term + traj_accel + kd_gains @ err_vel + kp_gains @ err_pos
        )
        curr_controls = np.clip(
            curr_controls_unsat,
            min_controls,
            max_controls,
        )

        if show_debug:
            print(f"curr_controls_unsat: {curr_controls_unsat}")
            print(f"curr_controls: {curr_controls}")

        # update the histories
        state_hist[i, :] = curr_state
        controls_hist[i, :] = curr_controls

        # advances the dynamics of the unicycle with the current controls
        curr_state = dyn_ext_unicycle_model_step(
            curr_t,
            dt,
            curr_state,
            curr_controls,
            max_v,
            max_omega,
        )
    return SimulationResult(state_hist, controls_hist)
