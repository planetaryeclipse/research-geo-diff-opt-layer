import numpy as np

from offline_data_gen.paths import (
    INDIV_INSTANCE_DIR,
    TRAIN_INSTANCE_PATH,
    VALID_INSTANCE_PATH,
)
from offline_data_gen.training_data import (
    TrainingInstance,
    aggregate_instances,
    randomize_instance,
    split_instances,
)

TRAIN_VALID_SPLIT = 0.7


def main():
    r = np.random.default_rng(44)

    instance_file_names = sorted(
        file.name
        for file in INDIV_INSTANCE_DIR.iterdir()
        if file.name is not ".gitkeep"
    )

    # aggegate all loaded instances and randomize
    all_instances = [
        TrainingInstance.load(INDIV_INSTANCE_DIR / instance)
        for instance in instance_file_names
    ]
    aggregated_data = aggregate_instances(all_instances)
    randomized_data = randomize_instance(aggregated_data, r)
    train_data, valid_data = split_instances(randomized_data, TRAIN_VALID_SPLIT)

    # saves to disk
    train_data.save(TRAIN_INSTANCE_PATH)
    valid_data.save(VALID_INSTANCE_PATH)


if __name__ == "__main__":
    main()
