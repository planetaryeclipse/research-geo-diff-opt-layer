import numpy as np

from offline_data_gen.episode_gen import DynUnicycleEpisode
from offline_data_gen.fb_linear_controller import simulate_under_fb_linear_control
from offline_data_gen.paths import CONTROLS_DIR, CONTROLS_FILE_PREFIX, EPISODES_DIR
from offline_data_gen.util import clean_dirs

KP_GAINS = 0.1 * np.eye(2)
KD_GAINS = 0.5 * np.eye(2)
OFFSET = 0.1


def main():
    episode_file_names = sorted(
        file.name for file in EPISODES_DIR.iterdir() if file.name is not ".gitkeep"
    )

    clean_dirs([CONTROLS_DIR])
    for i, file in enumerate(episode_file_names):
        episode = DynUnicycleEpisode.load(EPISODES_DIR / file)
        sim_result = simulate_under_fb_linear_control(
            episode, KP_GAINS, KD_GAINS, OFFSET
        )

        sim_result.save(CONTROLS_DIR / CONTROLS_FILE_PREFIX.format(id=i))


if __name__ == "__main__":
    main()
