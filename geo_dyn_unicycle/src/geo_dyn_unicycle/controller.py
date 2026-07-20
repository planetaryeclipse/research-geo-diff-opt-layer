import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(
        self,
        num_hidden_1=20,
        num_hidden_2=20,
    ):
        super().__init__()

        self._input_features = sum(
            [
                2,  # tracking position error
                3,  # other state variables
                2,  # trajectory velocity
                2,  # trajectory accel
            ]
        )
        self._output_features = 2  # controls
        self._cntrllr = nn.Sequential(
            nn.Linear(self._input_features, num_hidden_1),
            nn.Tanh(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.Tanh(),
            nn.Linear(num_hidden_2, self._output_features),
        )

    def forward(
        self,
        state: torch.Tensor,
        traj_pos: torch.Tensor,
        traj_vel: torch.Tensor,
        traj_acc: torch.Tensor,
    ):
        pos_state = state[:2]
        pos_err = traj_pos - pos_state  # prevents absolute position-based behavior
        non_pos_state = state[2:]

        is_batched = len(state.shape) > 1
        input = torch.concatenate(
            (pos_err, non_pos_state, traj_vel, traj_acc),
            dim=0 if not is_batched else 1,
        )
        out = self._cntrllr.forward(input)
        return out
