from geo_dyn_unicycle.model import ROBOMASTER_MAX_ANG_RATE, ROBOMASTER_MAX_SPEED
import numpy as np
import matplotlib.pyplot as plt

from offline_data_gen.fb_linear_controller import simulate_under_fb_linear_control
from offline_data_gen.episode_gen import DynUnicycleEpisode
from offline_data_gen.paths import EPISODES_DIR

episode = DynUnicycleEpisode.load(EPISODES_DIR / "episode_000.npz")
print(episode)

kp_gains = 0.4 * np.eye(2)
kd_gains = 0.1 * np.eye(2)
offset = 0.1

min_controls = np.array([-0.5, -0.2])
max_controls = np.array([0.5, 0.2])

sim_result = simulate_under_fb_linear_control(
    episode,
    kp_gains,
    kd_gains,
    offset,
    min_controls,
    max_controls,
    ROBOMASTER_MAX_SPEED,
    ROBOMASTER_MAX_ANG_RATE,
    show_debug=True,
)
