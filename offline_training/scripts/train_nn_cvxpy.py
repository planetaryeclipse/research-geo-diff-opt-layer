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
from offline_training.paths import CVXPY_RESULTS
from offline_training.results_setup import get_run_dir

CBF_K1 = 1.5
CBF_K2 = 1.0

NUM_HIDDEN_1 = 60
NUM_HIDDEN_2 = 60

NUM_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3

RNG_SEED = 42

CHKPT_INTERVAL = 10


def batch_loop_cvxpy(
    dataloader: DataLoader[DynUnicycleDataset],
    model: Controller,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    constr_layer: cvxpylayer.CvxpyLayer,
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

        batched_cbf_a, batched_cbf_b = batched_cbf_ko_coeffs(
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

        (u_safe,) = constr_layer(batched_cbf_a, batched_cbf_b, u_unsafe)

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
        "rng_seed": RNG_SEED,
        "lr": LR,
    }

    is_new_run_dir = args.resume is None

    run_dir: Path
    if is_new_run_dir:
        run_dir = get_run_dir(CVXPY_RESULTS)
    else:
        run_dir = CVXPY_RESULTS / args.resume
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

    # setup the cvxpy optimization problem
    u = cp.Variable(2)
    a_hocbf = cp.Parameter((1, 2))
    b_hocbf = cp.Parameter(1)
    u_nom = cp.Parameter(2)

    cbf_cost = 0.5 * cp.norm2(u - u_nom) ** 2
    cbf_constr = a_hocbf @ u + b_hocbf <= 0

    problem = cp.Problem(cp.Minimize(cbf_cost), [cbf_constr])
    cbf_layer = cvxpylayer.CvxpyLayer(
        problem,
        parameters=[a_hocbf, b_hocbf, u_nom],
        variables=[u],
    )

    # setup the training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=uni_cntrllr.parameters(), lr=LR)

    # run the training
    pbar = tqdm.tqdm(range(start_epoch, NUM_EPOCHS), desc="Training")
    for epoch in pbar:
        train_unsafe_loss, train_safe_loss = batch_loop_cvxpy(
            train_loader,
            uni_cntrllr,
            loss_fn,
            optimizer,
            cbf_layer,
            perform_train=True,
        )
        valid_unsafe_loss, valid_safe_loss = batch_loop_cvxpy(
            valid_loader,
            uni_cntrllr,
            loss_fn,
            optimizer,
            cbf_layer,
            perform_train=True,
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
