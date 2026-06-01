import numpy as np

from pathlib import Path
from typing import List, Tuple

from scipy.interpolate import CubicSpline
from dataclasses import dataclass


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


def generate_position_splines(waypoints: List[Tuple[np.ndarray, float]]) -> List[CubicSpline]:
    n = len(waypoints[0][0])

    waypoint_t = np.array(time for _, time in waypoints)
    waypoint_coords = [np.array(pos[i] for pos, _ in waypoints) for i in range(n)]
    waypoint_splines = [CubicSpline(waypoint_t, waypoint_coord) for waypoint_coord in waypoint_coords]

    return waypoint_splines


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    ddx: np.ndarray


def generate_trajectory(
    min_next_dist: float,
    max_next_dist: float,
    min_duration: float,
    max_duration: float,
    total_time: float,
    r: np.random.Generator,
    start_waypoint: np.ndarray = np.zeros(2),
    velocity_bias: float = 1.5,
    sample_time: float = 0.01,
) -> Trajectory:
    waypoints = generate_waypoints(
        min_next_dist, max_next_dist, min_duration, max_duration, total_time, start_waypoint, velocity_bias, r
    )
    waypoint_splines = generate_position_splines(waypoints)

    sample_times = np.arange(0.0, total_time + sample_time, sample_time)

    traj_x = [coord_spline(sample_times) for coord_spline in waypoint_splines]
    traj_dx = [coord_spline.derivative()(sample_times) for coord_spline in waypoint_splines]
    traj_ddx = [coord_spline.derivative(2)(sample_times) for coord_spline in waypoint_splines]

    return Trajectory(sample_times, np.stack(traj_x, axis=1), np.stack(traj_dx, axis=1), np.stack(traj_ddx, axis=1))


@dataclass
class Episode:
    trajectory: Trajectory
    start: np.ndarray

    def save(self, path: Path, **args):
        np.savez(
            path,
            t=self.trajectory.t,
            x=self.trajectory.x,
            dx=self.trajectory.dx,
            ddx=self.trajectory.ddx,
            start=self.start,
            *args,
        )

    @classmethod
    def load(cls, path: Path) -> Episode:
        data = np.load(path)

        traj = Trajectory(data["t"], data["x"], data["dx"], data["ddx"])
        start = data["start"]

        return Episode(traj, start)

    @classmethod
    def save_all(cls, dir: Path, episodes: List[Episode]):
        for i, episode in enumerate(episodes):
            file = dir / f"episode_{i:03d}"
            episode.save(file, id=i)


def generate_episodes(
    min_next_dist: float,
    max_next_dist: float,
    min_duration: float,
    max_duration: float,
    total_time: float,
    num_starts: int,
    start_var: float,
    r: np.random.Generator,
    start_waypoint: np.ndarray = np.zeros(2),
    velocity_bias: float = 1.5,
    sample_time: float = 0.01,
) -> List[Episode]:
    traj = generate_trajectory(
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
    true_start = traj.x[0, :]
    n = len(true_start)

    return [Episode(traj, r.multivariate_normal(true_start, start_var * np.eye(n))) for _ in range(num_starts)]
