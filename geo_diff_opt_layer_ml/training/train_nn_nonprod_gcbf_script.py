# %% [markdown]
# Train the network with the non-product formulation of the GCBF.


# %%
import torch

# torch.set_grad_enabled(True)  # prevent recompilation shenanigans
# torch._dynamo.config.recompile_limit = 100

import torch.nn as nn
import tqdm
import pathlib

import numpy as np

from diff_mfld_optim.geometry.metric import RnMetricField
from diff_mfld_optim.optim.constrained import (
    ConstrainedSolverCfg,
    ConstrainedSolverMethod,
)
from diff_mfld_optim.optim.subsolver import SolverCfg, SubsolverMethod

from datetime import datetime
from torch.utils.data import DataLoader
from multiprocessing.dummy import Pool

from controller import Controller
from diff_mfld_optim.mfld_util import MfldCfg

from geo_diff_opt_layer_ml.util.nominal_mpc_dataloader import (
    EPISODES_TRAIN_DIR,
    EPISODES_VALID_DIR,
    MPC_TRAIN_DIR,
    MPC_VALID_DIR,
    KO_TRAIN_DIR,
    KO_VALID_DIR,
    MPCEpisodeDataset,
)

from diff_mfld_optim.layers.diff_opt_layer import DiffMfldOptimLayer

from gcbf_nonprod import GCBF_Cost, GCBF_Constraint

# %%
import sys

# for efficiency in training given the cpu-bound process we need to use a
# version of Python with the gil disabled
sys._is_gil_enabled()

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
# setup the geometric optimization problem

gcbf_f = GCBF_Cost()
gcbf_g = GCBF_Constraint()


metric_field = RnMetricField(2)  # Euclidean for now
conn = metric_field.christoffels()  # Levi-Citivta connection
mfld_cfg = MfldCfg(metric_field, conn)

subsolver_cfg = SolverCfg()
constr_solv_cfg = ConstrainedSolverCfg(
    SubsolverMethod.RIEM_GRAD_DESCENT,
    subsolver_cfg,  # only one available (for now)
)

geo_constr_layer = DiffMfldOptimLayer(
    gcbf_f,
    [gcbf_g],
    [],
    mfld_cfg,
    constr_solv_cfg,
    ConstrainedSolverMethod.RALM,  # only one available (for now)
)
geo_constr_layer


# %%
def _filter_batch(p, x_traj, y_traj, u, ko):
    # filters the batch so that we don't train on any cases where the current
    # state of the system is inside the keep-out region (as we generated them
    # randomly and cannot do this directly inside the dataloader)

    x, y = p[:, 0], p[:, 1]
    ko_x, ko_y, ko_rad = ko[:, 0], ko[:, 1], ko[:, 6]

    # determines which bached states are outside thet keep-out radius
    dist_sqr = (x - ko_x) ** 2 + (y - ko_y) ** 2
    rad_sqr = ko_rad**2

    outside_ko_idxs = dist_sqr > rad_sqr

    # for testing purposes
    # outside_ko_idxs[4:] = torch.zeros(len(outside_ko_idxs) - 4, dtype=torch.bool)

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


def train_loop_gcbf(
    dataloader: DataLoader,
    model: Controller,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    geo_constr_layer: DiffMfldOptimLayer,
    k1: torch.Tensor,
    k2: torch.Tensor,
    pool: Pool,
    num_batches: int,  # manual limits on the number of batches
):
    model.train()

    batch_safe_loss = []
    batch_unsafe_loss = []
    for batch, (p, x_traj, y_traj, u, ko) in enumerate(dataloader):
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
        pred_u = model(p_filt, x_traj_filt, y_traj_filt)  # , h, h_grad)

        # uses the geometric safety layer to compute a safe input u
        safe_u = geo_constr_layer(
            pred_u,
            (pred_u, p_filt, ko_filt),  # batched arguments
            (k1, k2),  # non-batched arguments
            pool,  # for speed
        )

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

        if (batch + 1) >= num_batches:
            break

    # this will be our metric of performance
    avg_safe_batch_loss = sum(batch_safe_loss) / len(batch_safe_loss)
    avg_unsafe_batch_loss = sum(batch_unsafe_loss) / len(batch_unsafe_loss)
    return avg_safe_batch_loss, avg_unsafe_batch_loss


def valid_loop_gcbf(
    dataloader: DataLoader,
    model: Controller,
    loss_fn,
    geo_constr_layer: DiffMfldOptimLayer,
    k1: torch.Tensor,
    k2: torch.Tensor,
    pool: Pool,
    num_batches: int,  # manual limits on the number of batches
):
    model.eval()

    batch_safe_loss = []
    batch_unsafe_loss = []
    for batch, (p, x_traj, y_traj, u, ko) in enumerate(dataloader):
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
        pred_u = model(p_filt, x_traj_filt, y_traj_filt)  # , h, h_grad)

        # uses the geometric safety layer to compute a safe input u
        safe_u = geo_constr_layer(
            pred_u,
            (pred_u, p_filt, ko_filt),  # batched arguments
            (k1, k2),  # non-batched arguments
            pool,  # for speed
        )

        # computes the loss between the filtered nominal controller and the
        # safe value of u (as computed through the CBF-QP)+
        safe_loss = loss_fn(safe_u, u_filt)
        unsafe_loss = loss_fn(pred_u, u_filt)

        batch_safe_loss.append(safe_loss.item())
        batch_unsafe_loss.append(unsafe_loss.item())

        print("Validation: finished batch...")

        if (batch + 1) >= num_batches:
            break

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
    has_cbfs=False,  # True,
).to(device)

cntrllr_model

# %%

training_data_dir = pathlib.Path("geo_results/nonprod_flat_metric")

# load the weights and history from file if specified
# TODO: implement if needed
# warm_start_from = (
#     training_data_dir.joinpath("bkup_20_model_2026_02_26__22_20.pth"),
#     21,
# )
warm_start_from = None
epoch_start_idx = 0

if warm_start_from is not None:
    cntrllr_model.load_state_dict(torch.load(warm_start_from[0], weights_only=True))
    epoch_start_idx = warm_start_from[1]

# setup model saving

backup_epochs_freq = 5  # how many epochs to wait before saving model


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
batch_size = 64
lr = 1e-4


# will run with a small set of batches to demonstrate the learning
train_batch_limit = torch.inf  # 10
valid_batch_limit = torch.inf  # 5

# given large computational cost we will learn with a set number of batches
# that can be learned effectively to demonstrate learning the underlying
# controller through the optimization layer
mpc_train_loader = DataLoader(mpc_train_dataset, batch_size=batch_size, shuffle=True)
mpc_valid_loader = DataLoader(mpc_valid_dataset, batch_size=batch_size, shuffle=True)

# cbf params
k1 = 1.0
k2 = 1.0

# optimization params (note the large lagrange multiplier limits given the use
# of the higher order control barrier functions)

constr_solv_cfg.penalty = 1.0  # must be greater than 1 to grow
constr_solv_cfg.penalty_growth = 1.10  # must be greater than 1
constr_solv_cfg.ratio = 0.7
constr_solv_cfg.max_iters = 1000
constr_solv_cfg.conv_eps = 1e-2  # this is ignored by constrained solver control

constr_solv_max_mult = torch.inf  # 1_000_000.0
constr_solv_cfg.g_mult_clips = (-constr_solv_max_mult, constr_solv_max_mult)
constr_solv_cfg.h_mult_clips = (-constr_solv_max_mult, constr_solv_max_mult)

subsolver_cfg.damp = 0.1
subsolver_cfg.damp_growth = 0.95
subsolver_cfg.max_iters = 2000
subsolver_cfg.damp_clip = [1e-4, 1.0]

constr_solv_cfg.subsolver_acc_min = 1e-4
constr_solv_cfg.subsolver_acc = 1e-3  # starting
constr_solv_cfg.subsolver_acc_growth = 0.95

loss_fn = nn.MSELoss()
# optimizer = torch.optim.RMSprop(params=cntrllr_model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(params=cntrllr_model.parameters(), lr=lr)
optimizer = torch.optim.Adam(params=cntrllr_model.parameters(), lr=lr)

# loss training history
safe_train_loss_hist = []
unsafe_train_loss_hist = []

safe_valid_loss_hist = []
unsafe_valid_loss_hist = []

# train the model

pbar = tqdm.tqdm(range(epoch_start_idx, epochs), desc="Training")


# as we're passing batches to an optimization problem (that's unfortunately
# entirely written in Python) then for speed we need to create a thread
# pool for efficiency (adjust the number as appropriate for your system)

num_processes = 14
with Pool(processes=num_processes) as pool:
    for epoch in pbar:
        safe_train_loss, unsafe_train_loss = train_loop_gcbf(
            mpc_train_loader,
            cntrllr_model,
            loss_fn,
            optimizer,
            geo_constr_layer,
            k1,
            k2,
            pool,
            train_batch_limit,
        )
        safe_valid_loss, unsafe_valid_loss = valid_loop_gcbf(
            mpc_valid_loader,
            cntrllr_model,
            loss_fn,
            geo_constr_layer,
            k1,
            k2,
            pool,
            valid_batch_limit,
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

        # creates a backup of the current data just incase
        if epoch > 0 and epoch % backup_epochs_freq == 0:
            save_model_data(
                safe_train_loss_hist,
                unsafe_train_loss_hist,
                safe_valid_loss_hist,
                unsafe_valid_loss_hist,
                training_data_dir,
                f"bkup_{epoch}",
            )

# # %%
# # save the model with a unique timestamp (to prevent overwriting models)

save_model_data(
    safe_train_loss_hist,
    unsafe_train_loss_hist,
    safe_valid_loss_hist,
    unsafe_valid_loss_hist,
    training_data_dir,
    "final",
)
