# %% [markdown]
# Train the network with no CBF. Note that this is just a test to validate that all the dataset generation works properly. This also tests that if we don't provide the directories for the keep-out (KO) regions then the dataset when used in the loader will only output the required tensors without the currently enabled KO.

# %%
import torch
import torch.nn as nn
import tqdm
import pathlib

import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader

from dyn_unicycle.dyn_unicycle.controller import Controller

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from geo_diff_opt_layer_ml.util.nominal_mpc_dataloader import (
    EPISODES_TRAIN_DIR,
    EPISODES_VALID_DIR,
    MPC_TRAIN_DIR,
    MPC_VALID_DIR,
    KO_TRAIN_DIR,
    KO_VALID_DIR,
    MPCEpisodeDataset,
)

from dyn_unicycle.dyn_unicycle.euclid_cbf import batched_cbf_ko_coeffs

# %%
# device = torch.device(
#     "cuda:0"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
device = torch.device("cpu")
device

# %%
mpc_train_dataset = MPCEpisodeDataset(
    EPISODES_TRAIN_DIR, MPC_TRAIN_DIR, KO_TRAIN_DIR, device=device
)
mpc_valid_dataset = MPCEpisodeDataset(
    EPISODES_VALID_DIR, MPC_VALID_DIR, KO_VALID_DIR, device=device
)

# %%
# setup the optimization problems

u = cp.Variable(2)

a = cp.Parameter((1, 2))
b = cp.Parameter(1)
u_f_nom = cp.Parameter(2)

# define the CBF-QP cost and constraints
cost = 0.5 * cp.norm2(u - u_f_nom) ** 2
constr = a @ u + b >= 0

objective = cp.Minimize(cost)
problem = cp.Problem(objective, [constr])

# constructs the layer (outside of the controller) that we will use to to both
# enforce safe outputs of the controller while still permitting training
constr_layer = CvxpyLayer(problem, parameters=[a, b, u_f_nom], variables=[u]).to(device)


# %%
def _filter_batch(p, x_traj, y_traj, u, ko):
    # filters the batch so that we don't train on any cases where the current
    # state of the system is inside the keep-out region (as we generated them
    # randomly adn cannot do this directly inside the dataloader)

    x, y = p[:, 0], p[:, 1]
    ko_x, ko_y, ko_rad = ko[:, 0], ko[:, 1], ko[:, 6]

    # determines which bached states are outside thet keep-out radius
    dist_sqr = (x - ko_x) ** 2 + (y - ko_y) ** 2
    rad_sqr = ko_rad**2

    outside_ko_idxs = dist_sqr >= rad_sqr

    # now filter the batch based on the valid idxs
    return (
        p[outside_ko_idxs, :],
        x_traj[outside_ko_idxs],
        y_traj[outside_ko_idxs],
        u[outside_ko_idxs, :],
        ko[outside_ko_idxs, :],
    )


def _h_batch(p, x_traj, y_traj, u, ko):
    x, y = p[:, 0], p[:, 1]
    ko_x, ko_y, ko_rad = ko[:, 0], ko[:, 1], ko[:, 6]

    return torch.sqrt((ko_x - x) ** 2 + (ko_y - y) ** 2) - ko_rad


def _h_grad(p, x_traj, y_traj, u, ko):
    x, y = p[:, 0], p[:, 1]
    ko_x, ko_y, ko_rad = ko[:, 0], ko[:, 1], ko[:, 6]

    x_diff = -2 * (ko_x - x) / torch.sqrt((ko_x - x) ** 2 + (ko_y - y) ** 2)
    y_diff = -2 * (ko_y - y) / torch.sqrt((ko_x - x) ** 2 + (ko_y - y) ** 2)

    grad = torch.zeros((len(x), 2))
    grad[:, 0] = x_diff
    grad[:, 1] = y_diff

    return grad


def train_loop_euclid_cbf(
    dataloader: DataLoader,
    model: Controller,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    constr_layer: CvxpyLayer,
    k1: torch.Tensor,
    k2: torch.Tensor,
):
    model.train()

    batch_safe_loss = []
    batch_unsafe_loss = []
    for _batch, (p, x_traj, y_traj, u, ko) in enumerate(dataloader):
        # filters out all test cases where the current state is inside the
        # keep-out region (which would never occur in practice) as this
        # filtering operation cannot be done in the dataloader itself
        p_filt, x_traj_filt, y_traj_filt, u_filt, ko_filt = _filter_batch(
            p, x_traj, y_traj, u, ko
        )

        # CBF values
        h = _h_batch(p_filt, x_traj_filt, y_traj_filt, u_filt, ko_filt)
        h_grad = _h_grad(p_filt, x_traj_filt, y_traj_filt, u_filt, ko_filt)

        # makes a prediction on the control inputs which serves as the nominal
        # control input of the system
        pred_u = model(p_filt, x_traj_filt, y_traj_filt)

        # uses the safety layer to compute a safe input u
        cbf_a, cbf_b = batched_cbf_ko_coeffs(p_filt, ko_filt, k1, k2)
        (safe_u,) = constr_layer(cbf_a, cbf_b, pred_u)
        safe_u = safe_u.float()  # cvxpy internally uses float64

        # computes the loss between the filtered nominal controller and the
        # safe value of u (as computed through the CBF-QP)
        safe_loss = loss_fn(safe_u, u_filt)
        unsafe_loss = loss_fn(pred_u, u_filt)

        # backpropagation (through the optimization layer)
        safe_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_safe_loss.append(safe_loss.item())
        batch_unsafe_loss.append(unsafe_loss.item())

        print("Training: finished batch...")

    # this will be our metric of performance
    avg_safe_batch_loss = sum(batch_safe_loss) / len(batch_safe_loss)
    avg_unsafe_batch_loss = sum(batch_unsafe_loss) / len(batch_unsafe_loss)
    return avg_safe_batch_loss, avg_unsafe_batch_loss


def valid_loop_euclid_cbf(
    dataloader: DataLoader,
    model: Controller,
    loss_fn,
    constr_layer: CvxpyLayer,
    k1: torch.Tensor,
    k2: torch.Tensor,
):
    model.eval()

    batch_safe_loss = []
    batch_unsafe_loss = []
    for _batch, (p, x_traj, y_traj, u, ko) in enumerate(dataloader):
        # filters out all test cases where the current state is inside the
        # keep-out region (which would never occur in practice) as this
        # filtering operation cannot be done in the dataloader itself
        p_filt, x_traj_filt, y_traj_filt, u_filt, ko_filt = _filter_batch(
            p, x_traj, y_traj, u, ko
        )

        # CBF values
        h = _h_batch(p_filt, x_traj_filt, y_traj_filt, u_filt, ko_filt)
        h_grad = _h_grad(p_filt, x_traj_filt, y_traj_filt, u_filt, ko_filt)

        # makes a prediction on the control inputs which serves as the nominal
        # control input of the system
        pred_u = model(p_filt, x_traj_filt, y_traj_filt)

        # uses the safety layer to compute a safe input u
        cbf_a, cbf_b = batched_cbf_ko_coeffs(p_filt, ko_filt, k1, k2)
        (safe_u,) = constr_layer(cbf_a, cbf_b, pred_u)
        safe_u = safe_u.float()  # cvxpy internally uses float64

        # computes the loss between the filtered nominal controller and the
        # safe value of u (as computed through the CBF-QP)+
        safe_loss = loss_fn(safe_u, u_filt)
        unsafe_loss = loss_fn(pred_u, u_filt)

        batch_safe_loss.append(safe_loss.item())
        batch_unsafe_loss.append(unsafe_loss.item())

        print("Validation: finished batch...")

    # this will be our metric of performance
    avg_safe_batch_loss = sum(batch_safe_loss) / len(batch_safe_loss)
    avg_unsafe_batch_loss = sum(batch_unsafe_loss) / len(batch_unsafe_loss)
    return avg_safe_batch_loss, avg_unsafe_batch_loss


# %%
# controller params
state_dim = 5  # num states of dynamic unicycle
control_dim = 2  # num of inputs
traj_dim = 2  # only feeding in the trajectory position (not vel., accel.)

num_hidden_1 = 60
num_hidden_2 = 60

cntrllr_model = Controller(
    state_dim=state_dim,
    control_dim=control_dim,
    traj_dim=traj_dim,
    num_hidden_1=num_hidden_1,
    num_hidden_2=num_hidden_2,
    has_cbfs=False,  # only learning the underlying controller (cbf inputs are disabled)
).to(device)

cntrllr_model

# %%

training_data_dir = pathlib.Path("../results/geo_results/euclid")

# load the weights and history from file if specified
# TODO: implement if needed
warm_start_from = None

# setup model saving

backup_epochs_freq = 10  # how many epochs to wait before saving model


def save_model_data(
    safe_train_loss_hist,
    unsafe_train_loss_hist,
    safe_valid_loss_hist,
    unsafe_valid_loss_hist,
    training_data_dir: pathlib.Path,
    result_prefix="bkup",
):
    # allows us to save our data intermittantly just in case we have an error
    # that occurs (so we can warm start and finish training) or want to end
    # the training early

    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")

    model_path = training_data_dir.joinpath(f"{result_prefix}_model_{timestamp}.pth")
    torch.save(cntrllr_model.state_dict(), model_path)

    safe_train_loss_hist_arr = np.asarray(safe_train_loss_hist)
    unsafe_train_loss_hist_arr = np.asarray(unsafe_train_loss_hist)

    safe_valid_loss_hist_arr = np.asarray(safe_valid_loss_hist)
    unsafe_valid_loss_hist_arr = np.asarray(unsafe_valid_loss_hist)

    np.save(
        training_data_dir.joinpath(f"{result_prefix}_safe_train_loss_hist_{timestamp}"),
        safe_train_loss_hist_arr,
    )
    np.save(
        training_data_dir.joinpath(
            f"{result_prefix}_unsafe_train_loss_hist_{timestamp}"
        ),
        unsafe_train_loss_hist_arr,
    )
    np.save(
        training_data_dir.joinpath(f"{result_prefix}_safe_valid_loss_hist_{timestamp}"),
        safe_valid_loss_hist_arr,
    )
    np.save(
        training_data_dir.joinpath(
            f"{result_prefix}_unsafe_valid_loss_hist_{timestamp}"
        ),
        unsafe_valid_loss_hist_arr,
    )


# %%
# hyperparameters
epochs = 100
batch_size = 128
lr = 1e-5

mpc_train_loader = DataLoader(mpc_train_dataset, batch_size=batch_size, shuffle=True)
mpc_valid_loader = DataLoader(mpc_valid_dataset, batch_size=batch_size, shuffle=True)

# cbf params
k1 = 1.0
k2 = 1.0


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=cntrllr_model.parameters(), lr=lr)

# loss training history
safe_train_loss_hist = []
unsafe_train_loss_hist = []

safe_valid_loss_hist = []
unsafe_valid_loss_hist = []

# train the model

pbar = tqdm.tqdm(range(epochs), desc="Training")

for epoch in pbar:
    safe_train_loss, unsafe_train_loss = train_loop_euclid_cbf(
        mpc_train_loader, cntrllr_model, loss_fn, optimizer, constr_layer, k1, k2
    )
    safe_valid_loss, unsafe_valid_loss = valid_loop_euclid_cbf(
        mpc_valid_loader, cntrllr_model, loss_fn, constr_layer, k1, k2
    )
    pbar.set_postfix(
        safe_train_loss=safe_train_loss,
        unsafe_train_loss=unsafe_train_loss,
        safe_valid_loss=safe_valid_loss,
        unsafe_valid_loss=unsafe_valid_loss,
    )

    safe_train_loss_hist.append(safe_train_loss)
    unsafe_train_loss_hist.append(unsafe_train_loss)

    safe_valid_loss_hist.append(safe_valid_loss)
    unsafe_valid_loss_hist.append(unsafe_valid_loss)

    # creates a backup of the current data just in case
    if epoch > 0 and epoch % backup_epochs_freq == 0:
        save_model_data(
            safe_train_loss_hist,
            unsafe_train_loss_hist,
            safe_valid_loss_hist,
            unsafe_valid_loss_hist,
            training_data_dir,
            f"bkup_{epoch}",
        )

# %%
# save the model with a unique timestamp (to prevent overwriting models)

save_model_data(
    safe_train_loss_hist,
    unsafe_train_loss_hist,
    safe_valid_loss_hist,
    unsafe_valid_loss_hist,
    training_data_dir,
    "final",
)
