import torch
import torch.nn as nn


class Controller(nn.Module):
    def __init__(
        self,
        state_dim=5,
        control_dim=2,
        traj_dim=2,
        num_hidden_1=20,
        num_hidden_2=20,
        has_cbfs=False,
    ):
        super().__init__()

        self._state_dim = state_dim
        self._control_dim = control_dim
        self._traj_dim = traj_dim
        self._has_cbfs = has_cbfs

        # inputs to network are current state, current position of the
        # trajectory, and the cbf value and grad (if safety is enabled)
        self._num_input_features = state_dim + traj_dim + (1 + 2 if has_cbfs else 0)
        self._num_output_features = control_dim

        # main controller without the safety layer which will be applied for
        # the specific trial being used
        self.cntrllr = nn.Sequential(
            nn.Linear(self._num_input_features, num_hidden_1),
            nn.Tanh(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.Tanh(),
            nn.Linear(num_hidden_2, control_dim),
        )

    def forward(self, p, x_traj, y_traj, cbf_hs=None, cbf_hs_grad=None):
        # for simplicity we will simply handle safety by training to accomodate
        # the cbf that is most dangerous (so has the lowest value) if provided

        # reshape so we can concatenate properly
        x_traj = torch.reshape(x_traj, (len(x_traj), 1))
        y_traj = torch.reshape(y_traj, (len(y_traj), 1))

        if cbf_hs is not None and self._has_cbfs:
            cbf_hs = torch.reshape(cbf_hs, (len(cbf_hs), 1))
            input = torch.cat((p, x_traj, y_traj, cbf_hs, cbf_hs_grad), dim=1)
        elif cbf_hs is None and not self._has_cbfs:
            input = torch.cat((p, x_traj, y_traj), dim=1)
        else:
            raise ValueError(
                "invalid cbf input and cbf cfg: cbf_hs is None: "
                f"{cbf_hs is None}, has_cbfs: {self._has_cbfs}"
            )

        out = self.cntrllr.forward(input)
        return out
