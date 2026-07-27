from dmol.diff_mfld.connection.methods.geod_log_diff import LogMapCovarMethod
from dmol.diff_mfld.connection.methods.methods import (
    DistanceMethod,
    ExpMapMethod,
    GeodParlTransp,
    GeodParlTranspMethod,
    LogMapMethod,
)
from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.riemann import MetricField, MetricLambdaField
from dmol.optim.constr.ralm import ralm
from dmol.optim.unconstr.rtr import rtr
from dmol.torch.dmol import DiffMfldOptimProblem
from geo_dyn_unicycle.gcbf import batched_gcbf_constr, batched_gcbf_cost, gcbf_cost
from offline_training.metrics import MetricOption
import tqdm
import yaml
import csv
import shutil

from typing import Callable
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import cvxpy as cp
import cvxpylayers.torch.cvxpylayer as cvxpylayer

from torch.utils.data import DataLoader

from geo_dyn_unicycle.euclid_cbf import batched_cbf_ko_coeffs
from geo_dyn_unicycle.controller import Controller
from offline_data_gen.dataloader import DynUnicycleDataset
from offline_data_gen.paths import TRAIN_INSTANCE_PATH, VALID_INSTANCE_PATH
from offline_data_gen.training_data import TrainingInstance
from offline_training.paths import DMOL_RESULTS
from offline_training.results_setup import get_run_dir

CBF_K1 = 1.5
CBF_K2 = 1.0

NUM_HIDDEN_1 = 60
NUM_HIDDEN_2 = 60

NUM_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3

METRIC = MetricOption.EULER
RETR_METHOD = ExpMapMethod.APPROX_O2
LOG_MAP_METHOD = LogMapMethod.APPROX_O2
LOG_MAP_COVAR_METHOD = LogMapCovarMethod.APPROX_O2
DIST_METHOD = DistanceMethod.APPROX_O2
GEOD_PARL_TRANSP_METHOD = GeodParlTranspMethod.APPROX_O2

OPTIM_TOL = 1e-2
OPTIM_MAX_ITERS = 500

OPTIM_PENALTY_START = 0.1
OPTIM_PENALTY_GROWTH = 1.1
OPTIM_INEQ_MULT_START = 0.0
OPTIM_INEQ_MULTS_MIN = -torch.inf
OPTIM_INEQ_MULTS_MAX = torch.inf
OPTIM_EQ_MULT_START = 0.0
OPTIM_EQ_MULTS_MIN = -torch.inf
OPTIM_EQ_MULTS_MAX = torch.inf
OPTIM_SUBSOLVER_TOL_START = 1e-1
OPTIM_SUBSOLVER_TOL_MIN = 1e-2
OPTIM_SUBSOLVER_TOL_DECAY = 0.9
OPTIM_SUBSOLVER_MAX_ITERS = 200
OPTIM_RATIO = 0.8

OPTIM_RTR_RADIUS_MAX = 0.5
OPTIM_RTR_RADIUS_START = 0.1
OPTIM_RTR_QUALITY_STEP_THRESH = 0.15
OPTIM_RTR_DEFAULT_RETR_DAMP = 0.9

RNG_SEED = 42
CHKPT_INTERVAL = 10


def batch_loop_dmol[U: Manifold](
    dataloader: DataLoader[DynUnicycleDataset],
    model: Controller,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    metric: MetricField[U],
    perform_train: bool,
) -> tuple[float, float]:
    if perform_train:
        model.train()
    else:
        model.eval()

    num_batches = len(dataloader)

    running_unsafe_loss = 0.0
    running_safe_loss = 0.0

    pbar = tqdm.tqdm(dataloader, desc="Batches", leave=False)
    for batch_data in pbar:
        u_unsafe = model(
            batch_data["state"]["state"],
            batch_data["traj"]["pos"],
            batch_data["traj"]["vel"],
            batch_data["traj"]["accel"],
        )

        # due to manual creation of the cost and constraint scalar functions then they need to be handled independently
        # for each sample as part of the whole batch (likely api should be improved for "production" ready library)

        batched_costs = batched_gcbf_cost(
            u_unsafe,
            metric,
            LOG_MAP_METHOD,
            LOG_MAP_COVAR_METHOD,
        )
        batched_constrs = batched_gcbf_constr(
            metric.bundle.base,
            batch_data["state"]["state"],
            batch_data["zones"]["pos"],
            batch_data["zones"]["vel"],
            batch_data["zones"]["accel"],
            batch_data["zones"]["radius"],
            batch_data["zones"]["radius_vel"],
            batch_data["zones"]["radius_accel"],
            CBF_K1,
            CBF_K2,
        )

        u_safe = torch.zeros_like(u_unsafe)
        for sample_idx in range(u_safe.shape[0]):
            cost, constr = batched_costs[sample_idx], batched_constrs[sample_idx]
            u_safe[sample_idx, :] = DiffMfldOptimProblem.apply(
                u_unsafe[sample_idx, :],
                cost,
                (constr,),
                (),
                metric,
                ralm,  # solver method
                {
                    "subsolver_method": rtr,
                    "subsolver_args": {
                        "radius_max": OPTIM_RTR_RADIUS_MAX,
                        "radius_start": OPTIM_RTR_RADIUS_START,
                        "quality_step_thresh": OPTIM_RTR_QUALITY_STEP_THRESH,
                        "default_retr_damp": OPTIM_RTR_DEFAULT_RETR_DAMP,
                    },
                    "penalty_start": OPTIM_PENALTY_START,
                    "penalty_growth": OPTIM_PENALTY_GROWTH,
                    "ineq_mult_start": OPTIM_INEQ_MULT_START,
                    "ineq_mults_min": OPTIM_INEQ_MULTS_MIN,
                    "ineq_mults_max": OPTIM_INEQ_MULTS_MAX,
                    "eq_mult_start": OPTIM_EQ_MULT_START,
                    "eq_mults_min": OPTIM_EQ_MULTS_MIN,
                    "eq_mults_max": OPTIM_EQ_MULTS_MAX,
                    "subsolver_tol_start": OPTIM_SUBSOLVER_TOL_START,
                    "subsolver_tol_min": OPTIM_SUBSOLVER_TOL_MIN,
                    "subsolver_tol_decay": OPTIM_SUBSOLVER_TOL_DECAY,
                    "subsolver_max_iters": OPTIM_SUBSOLVER_MAX_ITERS,
                    "ratio": OPTIM_RATIO,
                },  # additional solver arguments
                None,  # use default levi-civita connection
                RETR_METHOD,
                DIST_METHOD,
                LOG_MAP_METHOD,
                GEOD_PARL_TRANSP_METHOD,
                OPTIM_TOL,
                OPTIM_MAX_ITERS,
            )

        loss_unsafe = loss_fn(batch_data["state"]["controls"], u_unsafe)
        loss_safe = loss_fn(batch_data["state"]["controls"], u_safe)

        # train through the optimization layer
        if perform_train:
            loss_safe.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_safe_loss += loss_safe.item()
        running_unsafe_loss += loss_unsafe.item()

    avg_unsafe_loss = running_unsafe_loss / num_batches
    avg_safe_loss = running_safe_loss / num_batches

    return avg_unsafe_loss, avg_safe_loss


def main():
    parser = ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume failed run")
    args = parser.parse_args()

    curr_config = {
        "cbf_k1": CBF_K1,
        "cbf_k2": CBF_K2,
        "num_hidden_1": NUM_HIDDEN_1,
        "num_hidden_2": NUM_HIDDEN_2,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "rng_seed": RNG_SEED,
        "metric": METRIC.name,
        "methods": {
            "retr_method": RETR_METHOD.name,
            "log_map_method": LOG_MAP_METHOD.name,
            "log_map_covar_method": LOG_MAP_COVAR_METHOD.name,
            "dist_method": DIST_METHOD.name,
            "geod_parl_transp_method": GEOD_PARL_TRANSP_METHOD.name,
        },
        "optim": {
            "tol": OPTIM_TOL,
            "max_iters": OPTIM_MAX_ITERS,
            "penalty_start": OPTIM_PENALTY_START,
            "penalty_growth": OPTIM_PENALTY_GROWTH,
            "ineq_mult_start": OPTIM_INEQ_MULT_START,
            "ineq_mults_min": OPTIM_INEQ_MULTS_MIN,
            "ineq_mults_max": OPTIM_INEQ_MULTS_MAX,
            "eq_mult_start": OPTIM_EQ_MULT_START,
            "eq_mults_min": OPTIM_EQ_MULTS_MIN,
            "eq_mults_max": OPTIM_EQ_MULTS_MAX,
            "subsolver_tol_start": OPTIM_SUBSOLVER_TOL_START,
            "subsolver_tol_min": OPTIM_SUBSOLVER_TOL_MIN,
            "subsolver_tol_decay": OPTIM_SUBSOLVER_TOL_DECAY,
            "subsolver_max_iters": OPTIM_SUBSOLVER_MAX_ITERS,
            "ratio": OPTIM_RATIO,
            "subsolver_args": {
                "rtr_radius_max": OPTIM_RTR_RADIUS_MAX,
                "rtr_radius_start": OPTIM_RTR_RADIUS_START,
                "rtr_quality_step_thresh": OPTIM_RTR_QUALITY_STEP_THRESH,
                "rtr_default_retr_damp": OPTIM_RTR_DEFAULT_RETR_DAMP,
            },
        },
    }

    is_new_run_dir = args.resume is None

    run_dir: Path
    if is_new_run_dir:
        run_dir = get_run_dir(DMOL_RESULTS)
    else:
        run_dir = DMOL_RESULTS / args.resume
        if not run_dir.exists():
            raise ValueError(f"run {run_dir} does not exist")

    config_path = run_dir / "config.yaml"
    history_path = run_dir / "history.csv"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"

    if is_new_run_dir:
        with open(config_path, "w") as file:
            yaml.safe_dump(curr_config, file)

        start_epoch = 0
        best_unsafe = torch.inf

        with open(history_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "train_unsafe_loss",
                    "train_safe_loss",
                    "valid_unsafe_loss",
                    "valid_safe_loss",
                    "lr",
                ]
            )
        torch.manual_seed(RNG_SEED)
        uni_cntrllr = Controller()
    else:
        with open(config_path, "r") as file:
            saved_config = yaml.safe_load(file)
        if saved_config != curr_config:
            raise ValueError("configuration from previous run differs")

        latest = torch.load(latest_path)
        start_epoch = latest["epoch"] + 1
        best_unsafe = latest["best_unsafe"]

        uni_cntrllr = Controller()
        uni_cntrllr.load_state_dict(latest["model"])
        torch.random.set_rng_state(latest["rng"])

    # loads the training/validation data
    train_data = DynUnicycleDataset(TrainingInstance.load(TRAIN_INSTANCE_PATH))
    valid_data = DynUnicycleDataset(TrainingInstance.load(VALID_INSTANCE_PATH))
    train_loader, valid_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
    ), DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
    )

    # setup the training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=uni_cntrllr.parameters(), lr=LR)

    U = Manifold[2]  # the input space
    metric = MetricLambdaField[U](METRIC)  # type: ignore

    # run the training
    pbar = tqdm.tqdm(range(start_epoch, NUM_EPOCHS), desc="Training")
    for epoch in pbar:
        train_unsafe_loss, train_safe_loss = batch_loop_dmol(
            train_loader,
            uni_cntrllr,
            loss_fn,
            optimizer,
            metric,
            perform_train=True,
        )
        valid_unsafe_loss, valid_safe_loss = batch_loop_dmol(
            valid_loader,
            uni_cntrllr,
            loss_fn,
            optimizer,
            metric,
            perform_train=False,
        )
        with open(history_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    train_unsafe_loss,
                    train_safe_loss,
                    valid_unsafe_loss,
                    valid_safe_loss,
                    LR,
                ]
            )

        # save the latest data and update the best model if applicable
        torch.save(
            {
                "epoch": epoch,
                "best_unsafe": min(valid_unsafe_loss, best_unsafe),
                "model": uni_cntrllr.state_dict(),
                "rng": torch.random.get_rng_state(),
            },
            latest_path,
        )
        if valid_unsafe_loss > best_unsafe:
            best_unsafe = valid_unsafe_loss
            shutil.copy2(latest_path, best_path)

        if epoch > 0 and epoch % CHKPT_INTERVAL == 0:
            shutil.copy2(latest_path, run_dir / f"chkpt_{epoch}.pt")

        pbar.set_postfix(
            train_unsafe=train_unsafe_loss,
            train_safe=train_safe_loss,
            valid_unsafe=valid_unsafe_loss,
            valid_safe=valid_safe_loss,
        )


if __name__ == "__main__":
    main()
