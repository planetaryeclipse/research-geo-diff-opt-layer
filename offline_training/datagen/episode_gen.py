import numpy as np

from pathlib import Path
from typing import List, Tuple

from scipy.interpolate import CubicSpline
from dataclasses import dataclass, asdict

from dacite import from_dict


def generate_waypoints(
    min_next_dist: float,
    max_next_dist: float,
    min_duration: float,
    max_duration: float,
    total_time: float,
    start_waypoint: np.ndarray,
    velocity_bias: float,
    r: np.random.Generator,
) -> List[Tuple[np.ndarray, float]]:
    curr_time = 0.0
    curr_waypoint = start_waypoint
    curr_vel = np.zeros_like(curr_waypoint)

    timestamped_waypoints = [(start_waypoint, 0.0)]
    while curr_time < total_time:
        # clamps time duration to ensure we don't cross the total time
        traversal_time = min_duration + (max_duration - min_duration) * r.random()
        next_time = np.minimum(curr_time + traversal_time, total_time)

        # gets a random direction vector
        traversal_ang = (2 * np.pi) * r.random()
        traversal_dir = np.array([np.cos(traversal_ang), np.sin(traversal_ang)])

        # to prevent a completely randomized path, in which direction reversals can cause sharp turns when being
        # transformed into a spline, we bias the direction of the next waypoint using the previous velocity
        bias = velocity_bias * curr_vel * traversal_time
        traversal_dist = min_next_dist + (max_next_dist - min_next_dist) * r.random()

        traversal_displacement = traversal_dist * traversal_dir + bias
        traversal_displacement /= np.linalg.norm(traversal_displacement)
        traversal_displacement *= traversal_dist

        next_waypoint = curr_waypoint + traversal_displacement
        timestamped_waypoints.append((next_waypoint, next_time))

        # updates counters to prepare for generation of the next node
        curr_time = next_time
        curr_waypoint = next_waypoint
        curr_vel = traversal_displacement / traversal_time

    return timestamped_waypoints


def generate_position_splines(
    waypoints: List[Tuple[np.ndarray, float]],
) -> List[CubicSpline]:
    n = len(waypoints[0][0])

    waypoint_t = np.array(time for _, time in waypoints)
    waypoint_coords = [np.array(pos[i] for pos, _ in waypoints) for i in range(n)]
    waypoint_splines = [
        CubicSpline(waypoint_t, waypoint_coord) for waypoint_coord in waypoint_coords
    ]

    return waypoint_splines


@dataclass
class DynUnicycleTrajectory:
    t: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    ddx: np.ndarray


def generate_dyn_unicycle_trajectory(
    min_next_dist: float,
    max_next_dist: float,
    min_duration: float,
    max_duration: float,
    total_time: float,
    start_waypoint: np.ndarray,
    velocity_bias: float,
    sample_time: float,
    r: np.random.Generator,
) -> DynUnicycleTrajectory:
    waypoints = generate_waypoints(
        min_next_dist,
        max_next_dist,
        min_duration,
        max_duration,
        total_time,
        start_waypoint,
        velocity_bias,
        r,
    )
    waypoint_splines = generate_position_splines(waypoints)

    sample_times = np.arange(0.0, total_time + sample_time, sample_time)

    traj_x = [coord_spline(sample_times) for coord_spline in waypoint_splines]
    traj_dx = [
        coord_spline.derivative()(sample_times) for coord_spline in waypoint_splines
    ]
    traj_ddx = [
        coord_spline.derivative(2)(sample_times) for coord_spline in waypoint_splines
    ]

    return DynUnicycleTrajectory(
        sample_times,
        np.stack(traj_x, axis=1),
        np.stack(traj_dx, axis=1),
        np.stack(traj_ddx, axis=1),
    )


@dataclass
class DynUnicycleEpisode:
    trajectory: DynUnicycleTrajectory
    start: np.ndarray

    def save(self, path: Path):
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> DynUnicycleEpisode:
        data = np.load(path)
        return from_dict(data_class=cls, data=data)


def wrap_ang(ang: float) -> float:
    is_pos_ang = np.sign(ang) == 1

    half_cycles = int(np.sign(ang) * ang / np.pi)
    on_pos_side = (half_cycles + (1 if not is_pos_ang else 0)) % 2 == 0

    ang_mag = np.abs(ang) % (2.0 * np.pi)

    if on_pos_side and is_pos_ang:
        return ang_mag
    elif not on_pos_side and is_pos_ang:
        return ang_mag - 2.0 * np.pi
    elif on_pos_side and not is_pos_ang:
        return 2.0 * np.pi - ang_mag
    else:
        # not on_pos_side and not is_pos_ang
        return -ang_mag


def generate_dyn_unicycle_episodes(
    min_next_dist: float,
    max_next_dist: float,
    min_duration: float,
    max_duration: float,
    total_time: float,
    num_starts: int,
    start_pos_var: float,
    start_ang_var: float,
    start_waypoint: np.ndarray,
    velocity_bias: float,
    sample_time: float,
    r: np.random.Generator,
) -> List[DynUnicycleEpisode]:
    traj = generate_dyn_unicycle_trajectory(
        min_next_dist,
        max_next_dist,
        min_duration,
        max_duration,
        total_time,
        r,
        start_waypoint,
        velocity_bias,
        sample_time,
    )
    true_start_pos = traj.x[0, :]
    n = len(true_start_pos)

    episodes = []
    for _ in range(num_starts):
        # generates a randomized starting position and orientation that is distributed around the direction vector
        # from the random start to the trajectory start
        start_pos = r.multivariate_normal(true_start_pos, start_pos_var * np.eye(n))
        start_pos_diff = start_pos - true_start_pos

        true_start_ang = np.atan2(start_pos_diff[0], start_pos_diff[1])
        start_ang = r.normal(true_start_ang, start_ang_var)

        # ensures the randomized angle is in the usual range of (-pi, pi)
        wrapped_ang = wrap_ang(start_ang)

        episodes.append(DynUnicycleEpisode(traj, np.array([*start_pos, wrapped_ang])))
    return episodes
