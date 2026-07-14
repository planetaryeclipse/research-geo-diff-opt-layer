import sys
import numpy as np

from dacite import from_dict

from dataclasses import dataclass, asdict
from pathlib import Path

from dyn_unicycle import dyn_ext_unicycle_model_step
from offline_training.data_gen.datagen.episode_gen import DynUnicycleEpisode


@dataclass
class SimulationResult:
    states: np.ndarray
    controls: np.ndarray

    def save(self, path: Path):
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> SimulationResult:
        data = np.load(path)
        return from_dict(data_class=cls, data=data)


def simulate_under_fb_linear_control(
    episode: DynUnicycleEpisode,
    kp_gains: np.ndarray,
    kd_gains: np.ndarray,
    offset: float,
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

    state_hist = np.zeros()
    controls_hist = np.zeros()

    num_timesteps = len(episode.trajectory.t)
    for i in range(num_timesteps):
        # gets the current state and trajectory state
        curr_t = episode.trajectory.t[i]

        curr_theta, curr_lin_vel, curr_ang_vel = curr_state[2:]
        curr_pos = curr_state[:2]
        curr_vel = curr_lin_vel * np.array([np.cos(curr_theta), np.sin(curr_theta)])

        traj_pos = episode.trajectory.x[i, :]
        traj_vel = episode.trajectory.dx[i, :]
        traj_accel = episode.trajectory.ddx[i, :]

        # generates inputs using feedback linearization scheme with a PD controller
        dyn_matrix = np.array(
            [
                [np.cos(curr_theta), -offset * np.sin(curr_theta)],
                [np.sin(curr_theta), offset * np.cos(curr_theta)],
            ]
        )
        resp_term = traj_accel
        resp_term += kd_gains @ (traj_vel - curr_vel)
        resp_term += kp_gains @ (traj_pos - curr_pos)
        resp_term += np.array(
            [
                curr_lin_vel * curr_ang_vel * np.sin(curr_theta),
                -curr_lin_vel * curr_ang_vel * np.cos(curr_theta),
            ]
        )
        resp_term += np.array(
            [
                offset * curr_lin_vel**2 * np.cos(curr_theta),
                -offset * curr_ang_vel**2 * np.sin(curr_theta),
            ]
        )
        curr_controls = np.linalg.inv(dyn_matrix) @ resp_term

        # update the histories
        state_hist[i, :] = curr_state
        controls_hist[i, :] = curr_controls

        # advances the dynamics of the unicycle with the current controls
        curr_state = dyn_ext_unicycle_model_step(curr_t, dt, curr_state, curr_controls)
    return SimulationResult(state_hist, controls_hist)
