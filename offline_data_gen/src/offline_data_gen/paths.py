import rootutils

_PROJ_ROOT = rootutils.setup_root(search_from=__file__, indicator=".project-root")

EPISODES_DIR = _PROJ_ROOT / "offline_data_gen/data/episodes"
EPISODE_FILE_PREFIX = "episode_{id:003d}.npz"

CONTROLS_DIR = _PROJ_ROOT / "offline_data_gen/data/controls"
CONTROLS_FILE_PREFIX = "controls_{id:003d}.npz"

INDIV_INSTANCE_DIR = _PROJ_ROOT / "offline_data_gen/data/instances/individual"
INDIV_INSTANCE_FILE_PREFIX = "instance_{id:003d}.npz"

TRAIN_INSTANCE_PATH = _PROJ_ROOT / "offline_data_gen/data/instances/training.npz"
VALID_INSTANCE_PATH = _PROJ_ROOT / "offline_data_gen/data/instances/validation.npz"
