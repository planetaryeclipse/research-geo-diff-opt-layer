import h5py

import numpy as np

from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from typing import Tuple, List


@dataclass
class MPCEpisode:
    # sample time for episode trajectory and mpc history
    sample_time: float

    # episode description
    start_pos: torch.Tensor  # starting position
    t_traj: torch.Tensor  # trajectory sample times
    x_traj: torch.Tensor  # trajectory x
    y_traj: torch.Tensor  # trajectory y
    dx_traj: torch.Tensor  # trajectory x velocity
    dy_traj: torch.Tensor  # trajectory y velocity
    ddx_traj: torch.Tensor  # trajectory x acceleration
    ddy_traj: torch.Tensor  # trajectory y acceleration

    # mpc description
    p_hist: torch.Tensor  # state history of model under mpc control
    u_hist: torch.Tensor  # control history of mpc
    traj_err_hist: torch.Tensor  # trajectory error history under mpc control

    # ko zones description (note that this is an optional parameter that if
    # defined will then be used to generate additional data augmentations)
    ko_x: torch.Tensor = None
    ko_y: torch.Tensor = None
    ko_vel_x: torch.Tensor = None
    ko_vel_y: torch.Tensor = None
    ko_accel_x: torch.Tensor = None
    ko_accel_y: torch.Tensor = None

    ko_radius: torch.Tensor = None
    ko_vel_radius: torch.Tensor = None
    ko_accel_radius: torch.Tensor = None


class MPCEpisodeDataset(Dataset):
    def __init__(self, episodes_dir: Path, mpc_dir: Path, ko_dir: Path = None, device=None):
        # checks the episode and mpc data are matching
        episode_files = [file.name for file in episodes_dir.iterdir() if file.name != ".gitkeep"]
        mpc_files = [file.name for file in mpc_dir.iterdir() if file.name != ".gitkeep"]

        ko_enabled = ko_dir is not None

        ko_files = [file.name for file in ko_dir.iterdir() if file.name != ".gitkeep"] if ko_enabled else None

        # ensures we have the same ordering (as iterating through a directory
        # does not provide any guarantees on the ordering
        episode_files.sort()
        mpc_files.sort()
        if ko_enabled:
            ko_files.sort()

        if (episode_files != mpc_files) or (ko_enabled and mpc_files != ko_files):
            raise ValueError(
                "mpc episode dataset has nonmatching files: " f"episodes_dir={episodes_dir}, mpc_dir={mpc_dir}"
            )

        # loads the data
        file_data: List[Tuple[str, MPCEpisode]] = []
        for filename in episode_files:
            episode_file = episodes_dir.joinpath(filename)
            mpc_file = mpc_dir.joinpath(filename)
            ko_file = ko_dir.joinpath(filename) if ko_dir is not None else None

            with (
                h5py.File(episode_file, "r") as episode_f,
                h5py.File(mpc_file, "r") as mpc_f,
            ):
                # the numpy data this was generated from are float64 but to be
                # compatible with the default torch layers then it must be float32
                episode = MPCEpisode(
                    torch.tensor(
                        episode_f.attrs["sample_time"],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.tensor(episode_f.attrs["start_pos"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["t_traj"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["x_traj"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["y_traj"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["dx_traj"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["dy_traj"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["ddx_traj"], dtype=torch.float32, device=device),
                    torch.tensor(episode_f["ddy_traj"], dtype=torch.float32, device=device),
                    torch.tensor(mpc_f["p_hist"], dtype=torch.float32, device=device),
                    torch.tensor(mpc_f["u_hist"], dtype=torch.float32, device=device),
                    torch.tensor(mpc_f["traj_err_hist"], dtype=torch.float32, device=device),
                )
            # if the keepout is defined then we add it to the episode
            # representation which we will use to generate additional data
            # permuations to handle training with cbfs
            if ko_enabled:
                with h5py.File(ko_file, "r") as ko_f:
                    episode.ko_x = torch.tensor(ko_f["ko_x"], dtype=torch.float32, device=device)
                    episode.ko_y = torch.tensor(ko_f["ko_y"], dtype=torch.float32, device=device)
                    episode.ko_vel_x = torch.tensor(ko_f["ko_vel_x"], dtype=torch.float32, device=device)
                    episode.ko_vel_y = torch.tensor(ko_f["ko_vel_y"], dtype=torch.float32, device=device)
                    episode.ko_accel_x = torch.tensor(ko_f["ko_accel_x"], dtype=torch.float32, device=device)
                    episode.ko_accel_y = torch.tensor(ko_f["ko_accel_y"], dtype=torch.float32, device=device)
                    episode.ko_radius = torch.tensor(ko_f["ko_radius"], dtype=torch.float32, device=device)
                    episode.ko_vel_radius = torch.tensor(ko_f["ko_vel_radius"], dtype=torch.float32, device=device)
                    episode.ko_accel_radius = torch.tensor(ko_f["ko_accel_radius"], dtype=torch.float32, device=device)
            file_data.append((filename, episode))
        self._file_data = file_data
        self._episode_len = len(file_data[0][1].t_traj)
        self._ko_enabled = ko_enabled
        self._ko_len = len(file_data[0][1].ko_x) if self._ko_enabled else None

    def __len__(self):
        # we are taking batches over all the episodes available
        total_steps_in_all_episodes = len(self._file_data) * self._episode_len
        return (
            total_steps_in_all_episodes
            if not self._ko_enabled
            # the same actions can be taken but now there are a number of
            # available keep out regionsd defined for each episode
            else total_steps_in_all_episodes * self._ko_len
        )

    def episode(self, idx):
        # gets the episode directly
        return self._file_data[idx]

    def __getitem__(self, idx):
        episode_idx = idx // (self._episode_len * self._ko_len) if self._ko_enabled else idx // self._episode_len
        _, episode = self._file_data[episode_idx]

        episode: MPCEpisode = episode  # for convenience

        if self._ko_enabled:
            in_episode_ko_idx = idx % (self._episode_len * self._ko_len)  # index of timestep and ko inside episode
            ko_idx = in_episode_ko_idx % self._ko_len  # index of the ko being used
            step_idx = in_episode_ko_idx // self._ko_len  # index of the step being used
        else:
            step_idx = idx % self._episode_len

        # gets the current values
        p_curr = episode.p_hist[step_idx, :]
        x_traj_curr = episode.x_traj[step_idx]
        y_traj_curr = episode.y_traj[step_idx]
        u_curr = episode.u_hist[step_idx, :]

        if self._ko_enabled:
            ko_x = episode.ko_x[ko_idx]
            ko_y = episode.ko_y[ko_idx]
            ko_vel_x = episode.ko_vel_x[ko_idx]
            ko_vel_y = episode.ko_vel_y[ko_idx]
            ko_accel_x = episode.ko_accel_x[ko_idx]
            ko_accel_y = episode.ko_accel_y[ko_idx]
            ko_radius = episode.ko_radius[ko_idx]
            ko_vel_radius = episode.ko_vel_radius[ko_idx]
            ko_accel_radius = episode.ko_accel_radius[ko_idx]

            # to prevent unwrapping nine new tensors when batching then we want
            # to concatenate them all for more convenient access
            ko = torch.hstack(
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
                )
            )
            return (p_curr, x_traj_curr, y_traj_curr, u_curr, ko)

        # if no keep outs are specified then default to the other data
        return p_curr, x_traj_curr, y_traj_curr, u_curr
