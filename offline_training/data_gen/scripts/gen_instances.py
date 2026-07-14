import sys
import numpy as np

from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent))

from offline_training.datagen.util.episode_gen import DynUnicycleEpisode
from offline_training.datagen.util.fb_linear_controller import SimulationResult
from offline_training.datagen.util.ko_zones import KeepOutZone, generate_keep_out_zones
from offline_training.datagen.util.training_data import generate_instances, TrainingInstance

from offline_training.datagen.util.util import (
    clean_dirs,
    EPISODES_DIR,
    CONTROLS_DIR,
    INDIV_INSTANCE_DIR,
    INDIV_INSTANCE_FILE_PREFIX,
)

NUM_ZONES = 10

POS_OFFSET_COVAR = 0.5 * np.eye(2)
POS_VEL_COVAR = 0.25 * np.eye(2)
POS_ACCEL_COVAR = 0.05 * np.eye(2)

RADIUS_MEAN = 0.5
RADIUS_STD = 0.2
RADIUS_VEL_STD = 0.1
RADIUS_ACCEL_STD = 0.05

MAX_ABOVE_KO_ZONE = 1.0
MIN_ABOVE_KO_ZONE = 0.0


def main():
    r = np.random.default_rng(43)

    episode_file_names = sorted(
        file.name for file in EPISODES_DIR.iterdir() if file.name is not ".gitkeep"
    )
    dyn_file_names = sorted(
        file.name for file in CONTROLS_DIR.iterdir() if file.name is not ".gitkeep"
    )

    all_instances: List[TrainingInstance] = []
    for episode_file, dyn_file in zip(episode_file_names, dyn_file_names):
        episode = DynUnicycleEpisode.load(episode_file)
        dyn_result = SimulationResult.load(dyn_file)
        ko_zones = generate_keep_out_zones(
            episode,
            NUM_ZONES,
            POS_OFFSET_COVAR,
            POS_VEL_COVAR,
            POS_ACCEL_COVAR,
            RADIUS_MEAN,
            RADIUS_STD,
            RADIUS_VEL_STD,
            RADIUS_ACCEL_STD,
            r,
        )

        instances = generate_instances(
            episode,
            ko_zones,
            dyn_result,
            max_above_ko_edge=MAX_ABOVE_KO_ZONE,
            min_above_ko_edge=MIN_ABOVE_KO_ZONE,
        )

        all_instances.extend(instances)

    # randomly shuffle the aarray

    clean_dirs([INDIV_INSTANCE_DIR])
    for i, instance in enumerate(all_instances):
        instance.save(INDIV_INSTANCE_DIR / INDIV_INSTANCE_FILE_PREFIX.format(id=i))


if __name__ == "__main__":
    main()
