import numpy as np

from dataclasses import dataclass
from typing import List

from offline_data_gen.episode_gen import DynUnicycleEpisode


@dataclass
class KeepOutZone:
    center: np.ndarray
    center_vel: np.ndarray
    center_accel: np.ndarray

    radius: float
    radius_vel: float
    radius_accel: float


def generate_keep_out_zones(
    episode: DynUnicycleEpisode,
    num_zones: int,
    pos_offset_covar: np.ndarray,
    pos_vel_covar: np.ndarray,
    pos_accel_covar: np.ndarray,
    radius_mean: float,
    radius_std: float,
    radius_vel_std: float,
    radius_accel_std: float,
    r: np.random.Generator,
) -> List[KeepOutZone]:
    # generates a number of random keep out zones specified by typical parameters but currently generates offsets to
    # be created off the path at occasional points alnog the trajectory of the obstacle

    zone_pos_offsets = r.multivariate_normal(np.zeros(2), pos_offset_covar, num_zones)
    zone_vels = r.multivariate_normal(np.zeros(2), pos_vel_covar, num_zones)
    zone_accels = r.multivariate_normal(np.zeros(2), pos_accel_covar, num_zones)

    zone_radii = np.clip(r.normal(radius_mean, radius_std, num_zones), 0.0, np.inf)
    zone_radii_vels = r.normal(0.0, radius_vel_std, num_zones)
    zone_radii_accels = r.normal(0.0, radius_accel_std, num_zones)

    # samples randomly from bins generated along the trajectory

    zones = []

    subinterval_samples = r.random(num_zones)
    num_idxs_in_episode = len(episode.trajectory.t)
    for interval_lb, interval_ub in zip(range(num_zones), range(1, num_zones + 1)):
        # boundary indices of the interval in the episode
        min_interval_idx = interval_lb / (num_zones + 1) * num_idxs_in_episode
        max_interval_idx = interval_ub / (num_zones + 1) * num_idxs_in_episode

        # creates an index inside the interval
        interval_sample_idx = int(
            min_interval_idx
            + subinterval_samples[interval_lb] * (max_interval_idx - min_interval_idx)
        )

        zone_pos = episode.trajectory.x[interval_sample_idx, :] + (
            zone_pos_offsets[interval_lb, :]
        )

        zones.append(
            KeepOutZone(
                center=zone_pos,
                center_vel=zone_vels[interval_lb, :],
                center_accel=zone_accels[interval_lb, :],
                radius=zone_radii[interval_lb],
                radius_vel=zone_radii_vels[interval_lb],
                radius_accel=zone_radii_accels[interval_lb],
            )
        )

    return zones
