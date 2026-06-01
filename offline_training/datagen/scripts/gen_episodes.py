import sys
import numpy as np

from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from offline_training.datagen.episode_gen import generate_episodes, Episode
from util import EPISODES_TRAIN_DIR, EPISODES_VALID_DIR

# trajectory generation parameters
START_WAYPOINT = np.zeros((2))
VELOCITY_BIAS = 1.5
SAMPLE_TIME = 0.01
MIN_NEXT_DIST = 3.0
MAX_NEXT_DIST = 5.0
MIN_DURATION = 3.0
MAX_DURATION = 5.0
TOTAL_TIME = 20.0

# episode generation parameters
NUM_TRAJECTORIES = 20
NUM_EPS_PER_TRAJECTORY = 5
EPISODE_START_VAR = 0.25

# training/validation data split
TRAIN_SPLIT = 0.7


def main():
    r = np.random.default_rng(42)

    # generates multiple randomized trajectories and generates episodes by varying the starting location
    all_episodes = []
    for _ in range(NUM_TRAJECTORIES):
        episodes = generate_episodes(
            min_next_dist=MIN_NEXT_DIST,
            max_next_dist=MAX_NEXT_DIST,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION,
            total_time=TOTAL_TIME,
            num_starts=NUM_EPS_PER_TRAJECTORY,
            start_var=EPISODE_START_VAR,
            r=r,
            start_waypoint=START_WAYPOINT,
            velocity_bias=VELOCITY_BIAS,
            sample_time=SAMPLE_TIME,
        )
        all_episodes.extend(episodes)

    # performs the training/validation splits
    num_train_episodes = int(TRAIN_SPLIT * len(all_episodes))
    train_episodes = [episode for i, episode in enumerate(all_episodes) if i <= num_train_episodes]
    valid_episodes = [episode for i, episode in enumerate(all_episodes) if i > num_train_episodes]

    Episode.save_all(EPISODES_TRAIN_DIR, train_episodes)
    Episode.save_all(EPISODES_VALID_DIR, valid_episodes)


if __name__ == "__main__":
    main()
