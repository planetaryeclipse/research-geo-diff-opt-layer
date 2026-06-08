from torch.utils.data import Dataset

from training_data import TrainingInstance


class DynUnicycleDataset(Dataset):

    def __init__(self, instance: TrainingInstance):
        self._instance = instance
        self._len = instance.traj.pos.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return {
            "traj": {
                "pos": self._instance.traj.pos[index],
                "vel": self._instance.traj.vel[index],
                "accel": self._instance.traj.accel[index],
            },
            "state": {
                "state": self._instance.state.state[index],
                "controls": self._instance.state.controls[index],
            },
            "zones": {
                "pos": self._instance.zones.pos[index],
                "vel": self._instance.zones.vel[index],
                "accel": self._instance.zones.accel[index],
                "radius": self._instance.zones.radius[index],
                "radius_vel": self._instance.zones.radius_vel[index],
                "radius_accel": self._instance.zones.radius_accel[index],
            },
        }

    def __getitems__(self, indices):
        self.__getitem__(indices)  # other method is directly compatible
