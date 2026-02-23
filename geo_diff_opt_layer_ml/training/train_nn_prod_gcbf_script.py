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
import matplotlib.pyplot as plt

from diff_mfld_optim.geometry.connection import Connection
from diff_mfld_optim.geometry.stspace_conn import StatespaceConnection
from diff_mfld_optim.geometry.joined_conn import JoinedConnection

from diff_mfld_optim.geometry.metric import MetricField, RnMetricField
from diff_mfld_optim.optim.constrained import (
    ConstrainedSolverCfg,
    ConstrainedSolverMethod,
    ConstrainedSolverResult,
)
from diff_mfld_optim.optim.subsolver import SolverCfg, SubsolverMethod

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from torch.utils.data import DataLoader

# import multiprocessing
# from multiprocessing.pool import Pool

# import pathos.multiprocessing
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing.dummy import Pool

# from pathos.multiprocessing import ProcessingPool

from controller import Controller
from diff_mfld_optim.optim.subsolver import OptimFunc, FuncArgs
from diff_mfld_optim.mfld_util import MfldCfg, dist_squared_map

from geo_diff_opt_layer_ml.util.nominal_mpc_dataloader import (
    EPISODES_TRAIN_DIR,
    EPISODES_VALID_DIR,
    MPC_TRAIN_DIR,
    MPC_VALID_DIR,
    KO_TRAIN_DIR,
    KO_VALID_DIR,
    MPCEpisode,
    MPCEpisodeDataset,
)

from gcbf import cbf_ko

from euclid_cbf import batched_cbf_ko_coeffs

from diff_mfld_optim.layers.diff_opt_layer import DiffMfldOptimLayer
from diff_mfld_optim.layers.prod_diff_opt_layer import ProdDiffMfldOptimLayer

# %%
import sys

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

state_dim = 5  # num states of dynamic unicycle
control_dim = 2  # num of inputs
ist_dim = state_dim + control_dim

# define the statespace system noting that we assume the base space to be the
# product manifold M x U (although it doesn't matter for this subconn)


def unicycle_f_stspace(p: torch.Tensor) -> torch.Tensor:
    state, cntrl = p[:state_dim], p[state_dim:]

    x, y, theta, v, omega = state[0], state[1], state[2], state[3], state[4]
    u_f, u_t = cntrl[0], cntrl[1]

    return torch.tensor([v * torch.cos(theta), v * torch.sin(theta), omega, u_f, u_t])


stspace_conn = StatespaceConnection(state_dim, control_dim, unicycle_f_stspace)

# define the riemannian geometry on the u space and note that this is similar
# to the setup used for the non product implementation

u_metric_field = RnMetricField(2)  # Euclidean for now
u_conn = u_metric_field.christoffels()  # Levi-Civita connection
u_mfld_cfg = MfldCfg(u_metric_field, u_conn)

# define the joined construction for use in the product optimization scheme

# NOTE: for optimization purposes with the product manifold formulation we need
# the first manifold to be the manifold of optimization so in our case U x M.
# However, we've defined our statespace to be the opposite as M x U as this is
# more natural for dynamics. So we'll demonstrate using a re-indexing when
# defining this joined connection

prod_optim_joined_conn = JoinedConnection(
    state_dim + control_dim,
    [
        (
            stspace_conn,
            (
                torch.arange(state_dim, ist_dim),  # state as second arg in product
                torch.arange(state_dim, ist_dim),  # state as second arg in product
                torch.concat(
                    (torch.arange(state_dim, ist_dim), torch.arange(state_dim))
                ),  # state as second, input as first
            ),
        ),
        (
            u_conn,
            (
                torch.arange(control_dim),
                torch.arange(control_dim),
                torch.arange(control_dim),
            ),
        ),
    ],
)

# setup the geometric optimization problem

# NOTE: the arguments are consistent with the product for the optimization
# problem formulation in that it is on U x M


def prod_gcbf_f(
    u: torch.Tensor, p: torch.Tensor, mfld_cfg: MfldCfg, u_nom: torch.Tensor, *args
):
    return 0.5 * dist_squared_map(u, u_nom, mfld_cfg)


def prod_gcbf_g(
    u: torch.Tensor,
    p: torch.Tensor,
    _mfld_cfg: MfldCfg,
    _u_nom: torch.Tensor,
    ko: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
):
    return cbf_ko(p, u, ko, k1, k2)


# optimization parameters

subsolver_cfg = SolverCfg()
constr_solv_cfg = ConstrainedSolverCfg(
    SubsolverMethod.RIEM_GRAD_DESCENT,
    subsolver_cfg,  # only one available (for now)
)


prod_geo_constr_layer = ProdDiffMfldOptimLayer(
    prod_gcbf_f,
    [prod_gcbf_g],
    [],
    u_mfld_cfg,
    prod_optim_joined_conn,
    constr_solv_cfg,
    ConstrainedSolverMethod.RALM,  # only one available (for now)
)

prod_geo_constr_layer


# # %%
def _filter_batch(p, x_traj, y_traj, u, ko):
    # filters the batch so that we don't train on any cases where the current
    # state of the system is inside the keep-out region (as we generated them
    # randomly and cannot do this directly inside the dataloader)

    x, y = p[:, 0], p[:, 1]
    ko_x, ko_y, ko_rad = ko[:, 0], ko[:, 1], ko[:, 6]

    # determines which bached states are outside thet keep-out radius
    dist_sqr = (x - ko_x) ** 2 + (y - ko_y) ** 2
    rad_sqr = ko_rad**2

    outside_ko_idxs = dist_sqr >= rad_sqr

    # for testing purposes
    # outside_ko_idxs[:] = torch.zeros(len(outside_ko_idxs) - 4, dtype=torch.bool)

    # now filter teh batch based on the valid idxs
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
    prod_geo_constr_layer: ProdDiffMfldOptimLayer,
    k1: torch.Tensor,
    k2: torch.Tensor,
    pool: Pool,
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
        pred_u = model(p_filt, x_traj_filt, y_traj_filt, h, h_grad)

        # uses the geometric safety layer to compute a safe input u
        safe_u = prod_geo_constr_layer(
            pred_u,
            p_filt,
            (pred_u, ko_filt),  # batched arguments
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

        break

    # this will be our metric of performance
    avg_safe_batch_loss = sum(batch_safe_loss) / len(batch_safe_loss)
    avg_unsafe_batch_loss = sum(batch_unsafe_loss) / len(batch_unsafe_loss)
    return avg_safe_batch_loss, avg_unsafe_batch_loss


def valid_loop_gcbf(
    dataloader: DataLoader,
    model: Controller,
    loss_fn,
    prod_geo_constr_layer: ProdDiffMfldOptimLayer,
    k1: torch.Tensor,
    k2: torch.Tensor,
    pool: Pool,
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
        pred_u = model(p_filt, x_traj_filt, y_traj_filt, h, h_grad)

        # uses the geometric safety layer to compute a safe input u
        safe_u = prod_geo_constr_layer(
            pred_u,
            p_filt,
            (pred_u, ko_filt),  # batched arguments
            (k1, k2),  # non-batched arguments
            pool,  # for speed
        )

        # computes the loss between the filtered nominal controller and the
        # safe value of u (as computed through the CBF-QP)+
        safe_loss = loss_fn(safe_u, u_filt)
        unsafe_loss = loss_fn(pred_u, u_filt)

        batch_safe_loss.append(safe_loss.item())
        batch_unsafe_loss.append(unsafe_loss.item())

        break

    # this will be our metric of performance
    avg_safe_batch_loss = sum(batch_safe_loss) / len(batch_safe_loss)
    avg_unsafe_batch_loss = sum(batch_unsafe_loss) / len(batch_unsafe_loss)
    return avg_safe_batch_loss, avg_unsafe_batch_loss


# %%
# controller params
traj_dim = 2  # only feeding in the trajectory position (not vel., accel.)

num_hidden_1 = 60
num_hidden_2 = 60

cntrllr_model = Controller(
    state_dim=state_dim,
    control_dim=control_dim,
    traj_dim=traj_dim,
    num_hidden_1=num_hidden_1,
    num_hidden_2=num_hidden_2,
    has_cbfs=True,
).to(device)

cntrllr_model

# %%
# hyperparameters
epochs = 150
batch_size = 128
lr = 0.001

mpc_train_loader = DataLoader(mpc_train_dataset, batch_size=batch_size, shuffle=True)
mpc_valid_loader = DataLoader(mpc_valid_dataset, batch_size=batch_size, shuffle=True)

# cbf params
k1 = 0.1
k2 = 0.05

# optimization params

constr_solv_cfg.penalty = 10.0  # must be greater than 1 to grow
constr_solv_cfg.penalty_growth = 1.1  # must be greater than 1
constr_solv_cfg.ratio = 0.2
constr_solv_cfg.max_iters = 1000
constr_solv_cfg.conv_eps = 1e-2  # this is ignored by constrained solver control
constr_solv_cfg.g_mult_clips = (-1000, 1000)
constr_solv_cfg.h_mult_clips = (-1000, 1000)

# subsolver_cfg.conv_eps = 1e-3
subsolver_cfg.damp = 0.9
subsolver_cfg.damp_growth = 0.95
subsolver_cfg.max_iters = 1000
subsolver_cfg.damp_clip = [1e-5, 1.0]

constr_solv_cfg.subsolver_acc_min = 1e-2
constr_solv_cfg.subsolver_acc = 1e-1  # starting
constr_solv_cfg.subsolver_acc_growth = 0.7


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=cntrllr_model.parameters(), lr=lr)

# train the model

pbar = tqdm.tqdm(range(epochs), desc="Training")

# loss training history
safe_train_loss_hist = []
unsafe_train_loss_hist = []

safe_valid_loss_hist = []
unsafe_valid_loss_hist = []

# as we're passing batches to an optimization problem (that's unfortunately
# entirely written in Python) then for speed we need to create a
# multiprocessing context to use as many cores as possible

num_processes = 28
with Pool(processes=num_processes) as pool:
    for epoch in pbar:
        safe_train_loss, unsafe_train_loss = train_loop_gcbf(
            mpc_train_loader,
            cntrllr_model,
            loss_fn,
            optimizer,
            prod_geo_constr_layer,
            k1,
            k2,
            pool,
        )
        safe_valid_loss, unsafe_valid_loss = valid_loop_gcbf(
            mpc_valid_loader,
            cntrllr_model,
            loss_fn,
            prod_geo_constr_layer,
            k1,
            k2,
            pool,
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

        print("Completed a full loop!")

# print(safe_train_loss)
# print(unsafe_train_loss)
# print(safe_valid_loss)
# print(unsafe_valid_loss_hist)

# # %%
safe_train_loss_hist = np.asarray(safe_train_loss_hist)
unsafe_train_loss_hist = np.asarray(unsafe_train_loss_hist)

safe_valid_loss_hist = np.asarray(safe_valid_loss_hist)
unsafe_valid_loss_hist = np.asarray(unsafe_valid_loss_hist)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# # plt.plot(safe_train_loss_hist)

# ax1.plot(safe_train_loss_hist)
# ax1.plot(safe_valid_loss_hist)

# ax2.plot(unsafe_train_loss_hist)
# ax2.plot(unsafe_valid_loss_hist)


# # %%
# # save the model with a unique timestamp (to prevent overwriting models)

output_dir = pathlib.Path("nn_with_prod_gcbf_layer")
timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")

model_path = output_dir.joinpath(f"model_{timestamp}.pth")
torch.save(cntrllr_model.state_dict(), model_path)

np.save(output_dir.joinpath(f"safe_train_loss_hist_{timestamp}"), safe_train_loss_hist)
np.save(
    output_dir.joinpath(f"unsafe_train_loss_hist_{timestamp}"), unsafe_train_loss_hist
)
np.save(output_dir.joinpath(f"safe_valid_loss_hist_{timestamp}"), safe_valid_loss_hist)
np.save(
    output_dir.joinpath(f"unsafe_valid_loss_hist_{timestamp}"), unsafe_valid_loss_hist
)
