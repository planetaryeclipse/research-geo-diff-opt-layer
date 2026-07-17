import numpy as np

from typing import List

from offline_data_gen.episode_gen import (
    DynUnicycleEpisode,
    generate_dyn_unicycle_episodes,
)
from offline_data_gen.paths import EPISODE_FILE_PREFIX, EPISODES_DIR
from offline_data_gen.util import clean_dirs

# trajectory generation parameters
START_WAYPOINT = np.zeros((2))
VELOCITY_BIAS = 1.5
SAMPLE_TIME = 0.01
MIN_NEXT_DIST = 1.0
MAX_NEXT_DIST = 2.0
MIN_DURATION = 3.0
MAX_DURATION = 5.0
TOTAL_TIME = 20.0

# episode generation parameters
NUM_TRAJECTORIES = 20
NUM_EPS_PER_TRAJECTORY = 5
START_POS_VAR = 1.0
START_ANG_VAR = np.pi


def main():
    r = np.random.default_rng(42)

    # generates multiple randomized trajectories and generates episodes by varying the starting location
    all_episodes: List[DynUnicycleEpisode] = []
    for _ in range(NUM_TRAJECTORIES):
        episodes = generate_dyn_unicycle_episodes(
            min_next_dist=MIN_NEXT_DIST,
            max_next_dist=MAX_NEXT_DIST,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION,
            total_time=TOTAL_TIME,
            num_starts=NUM_EPS_PER_TRAJECTORY,
            start_pos_var=START_POS_VAR,
            start_ang_var=START_ANG_VAR,
            start_waypoint=START_WAYPOINT,
            velocity_bias=VELOCITY_BIAS,
            sample_time=SAMPLE_TIME,
            r=r,
        )
        all_episodes.extend(episodes)

    clean_dirs([EPISODES_DIR])
    for i, episode in enumerate(all_episodes):
        episode.save(EPISODES_DIR / EPISODE_FILE_PREFIX.format(id=i))


if __name__ == "__main__":
    main()
