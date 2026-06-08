import numpy as np

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

from dacite import from_dict

import episode_gen
import ko_zones
import fb_linear_controller


@dataclass
class Trajectory:
    pos: np.ndarray
    vel: np.ndarray
    accel: np.ndarray


@dataclass
class KeepOutZones:
    pos: np.ndarray
    vel: np.ndarray
    accel: np.ndarray

    radius: np.ndarray
    radius_vel: np.ndarray
    radius_accel: np.ndarray


@dataclass
class ControllerState:
    state: np.ndarray
    controls: np.ndarray


@dataclass
class TrainingInstance:
    traj: Trajectory
    zones: KeepOutZones
    state: ControllerState

    def save(self, path: Path):
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> TrainingInstance:
        data = np.load(path)
        return from_dict(data_class=cls, data=data)


def generate_instance(
    episode: episode_gen.DynUnicycleEpisode,
    zone: ko_zones.KeepOutZone,
    sim: fb_linear_controller.SimulationResult,
    max_above_ko_edge: float,
    min_above_ko_edge: float,
) -> TrainingInstance:
    sample_len = len(episode.trajectory.t)

    # to prevent training from being overly dominated by data obtained far from the keep-out zone (where the correction
    # from the GHOCBF is relatively minimal) then we only permit a band just outside the edge of the provided zone

    valid_idxs = []
    for i in range(sample_len):
        state_pos = sim.states[i, :2]
        dist_above_ko_edge = np.linalg.norm(state_pos - zone.center) - zone.radius
        if (
            dist_above_ko_edge > max_above_ko_edge
            or dist_above_ko_edge < min_above_ko_edge
        ):
            continue

        valid_idxs.append(i)

    # builds up an instance containing the desired trajectory, keep-out zone information, and state/controls from the
    # simulation at each timestep for future use in training

    return TrainingInstance(
        Trajectory(
            pos=episode.trajectory.x[valid_idxs, :],
            vel=episode.trajectory.dx[valid_idxs, :],
            accel=episode.trajectory.ddx[valid_idxs, :],
        ),
        KeepOutZones(
            pos=np.repeat(zone.center[:, np.newaxis], sample_len, axis=1),
            vel=np.repeat(zone.center_vel[:, np.newaxis], sample_len, axis=1),
            accel=np.repeat(zone.center_accel[:, np.newaxis], sample_len, axis=1),
            radius=zone.radius * np.ones(sample_len),
            radius_vel=zone.radius_vel * np.ones(sample_len),
            radius_accel=zone.radius_accel * np.ones(sample_len),
        ),
        ControllerState(
            state=sim.states[valid_idxs, :], controls=sim.controls[valid_idxs, :]
        ),
    )


def generate_instances(
    episode: episode_gen.DynUnicycleEpisode,
    zones: List[ko_zones.KeepOutZone],
    sim: fb_linear_controller.SimulationResult,
    max_above_ko_edge: float,
    min_above_ko_edge: float,
) -> List[TrainingInstance]:
    return [
        generate_instance(episode, zone, sim, max_above_ko_edge, min_above_ko_edge)
        for zone in zones
    ]


def aggregate_instances(instances: List[TrainingInstance]) -> TrainingInstance:
    return TrainingInstance(
        Trajectory(
            pos=np.stack([instance.traj.pos for instance in instances]),
            vel=np.stack([instance.traj.vel for instance in instances]),
            accel=np.stack([instance.traj.accel for instance in instances]),
        ),
        KeepOutZones(
            pos=np.stack([instance.zones.pos for instance in instances]),
            vel=np.stack([instance.zones.vel for instance in instances]),
            accel=np.stack([instance.zones.accel for instance in instances]),
            radius=np.stack([instance.zones.radius for instance in instances]),
            radius_vel=np.stack([instance.zones.radius_vel for instance in instances]),
            radius_accel=np.stack(
                [instance.zones.radius_accel for instance in instances]
            ),
        ),
        ControllerState(
            state=np.stack([instance.state.state for instance in instances]),
            controls=np.stack([instance.state.controls for instance in instances]),
        ),
    )


def randomize_instance(instance: TrainingInstance, r: np.random.Generator):
    num_samples = instance.traj.pos.shape[0]
    idxs = r.shuffle(np.arange(num_samples))

    return TrainingInstance(
        Trajectory(
            pos=instance.traj.pos[idxs, :],
            vel=instance.traj.vel[idxs, :],
            accel=instance.traj.accel[idxs, :],
        ),
        KeepOutZones(
            pos=instance.zones.pos[idxs, :],
            vel=instance.zones.vel[idxs, :],
            accel=instance.zones.accel[idxs, :],
            radius=instance.zones.radius[idxs],
            radius_vel=instance.zones.radius_vel[idxs],
            radius_accel=instance.zones.radius_accel[idxs],
        ),
        ControllerState(
            state=instance.state.state[idxs, :],
            controls=instance.state.controls[idxs, :],
        ),
    )


def split_instances(
    instance: TrainingInstance, split_percent: float = 0.7
) -> Tuple[TrainingInstance, TrainingInstance]:

    num_samples = instance.traj.pos.shape[0]
    num_train_samples = int(num_samples * split_percent)

    train_instance = TrainingInstance(
        Trajectory(
            pos=instance.traj.pos[:num_train_samples, :],
            vel=instance.traj.vel[:num_train_samples, :],
            accel=instance.traj.accel[:num_train_samples, :],
        ),
        KeepOutZones(
            pos=instance.zones.pos[:num_samples, :],
            vel=instance.zones.vel[:num_samples, :],
            accel=instance.zones.accel[:num_samples, :],
            radius=instance.zones.radius[:num_samples],
            radius_vel=instance.zones.radius_vel[:num_samples],
            radius_accel=instance.zones.radius_accel[:num_samples],
        ),
        ControllerState(
            state=instance.state.state[:num_samples, :],
            controls=instance.state.controls[:num_samples, :],
        ),
    )

    valid_instance = TrainingInstance(
        Trajectory(
            pos=instance.traj.pos[num_train_samples:, :],
            vel=instance.traj.vel[num_train_samples:, :],
            accel=instance.traj.accel[num_train_samples:, :],
        ),
        KeepOutZones(
            pos=instance.zones.pos[num_samples:, :],
            vel=instance.zones.vel[num_samples:, :],
            accel=instance.zones.accel[num_samples:, :],
            radius=instance.zones.radius[num_samples:],
            radius_vel=instance.zones.radius_vel[num_samples:],
            radius_accel=instance.zones.radius_accel[num_samples:],
        ),
        ControllerState(
            state=instance.state.state[num_samples:, :],
            controls=instance.state.controls[num_samples:, :],
        ),
    )

    return train_instance, valid_instance
