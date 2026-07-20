from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
import cattrs
import rootutils
import torch
import numpy as np


from external.dmol.src.dmol.diff_mfld.connection.methods.geod_log_diff import (
    LogMapCovarMethod,
)
from external.dmol.src.dmol.diff_mfld.connection.methods.methods import (
    Distance,
    DistanceMethod,
    ExpMapMethod,
    LogMapMethod,
)
from external.dmol.src.dmol.diff_mfld.field.riem_fields import RiemSqrDist
from external.dmol.src.dmol.diff_mfld.field.util import coord_repr
from external.dmol.src.dmol.diff_mfld.mfld import Manifold
from external.dmol.src.dmol.diff_mfld.riemann import MetricLambdaField
from external.dmol.src.dmol.optim.constr.ralm import ralm
from external.dmol.src.dmol.optim.methods import ConstrOptimFn, Retraction
from external.dmol.src.dmol.optim.unconstr.rgd import rgd
from external.dmol.src.dmol.optim.unconstr.rtr import rtr

# solver args
TOL = 1e-3
MAX_ITERS = 200
METHODS_APPROX_O1 = (
    ExpMapMethod.APPROX_O1,
    DistanceMethod.APPROX_O1,
    LogMapMethod.APPROX_O1,
    LogMapCovarMethod.APPROX_O1,
)
METHODS_APPROX_O2 = (
    ExpMapMethod.APPROX_O2,
    DistanceMethod.APPROX_O2,
    LogMapMethod.APPROX_O2,
    LogMapCovarMethod.APPROX_O2,
)
METHODS_APPROX_O3 = (
    ExpMapMethod.APPROX_O3,
    DistanceMethod.APPROX_O3,
    LogMapMethod.APPROX_O3,
    LogMapCovarMethod.APPROX_O3,
)
METHODS_APPROX_O4 = (
    ExpMapMethod.APPROX_O4,
    DistanceMethod.APPROX_O4,
    LogMapMethod.APPROX_O4,
    LogMapCovarMethod.APPROX_O4,
)

# problem setup
COORD_SCALING_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.50]
COST_SCALING_FACTOR = 0.02
CONSTR_SCALING_FACTOR = 0.02
COST_CENTER = np.array([-5.0, -1.0, 3.0])
CONSTR_CENTER = np.array([-2.0, 3.0, 1.0])
CONSTR_RADIUS = 3.0
START_POS_MEAN = np.array([2.0, -5.0, -8.0])
START_POS_COVAR = 2.0**2 * np.eye(3)
NUM_STARTS = 20
RNG_SEED = 42

# ralm specific arguments
SUBSOLVER_RGD = (rgd, {"damp": 0.15})
IDENT_SYMM_OPER = lambda p: p
SUBSOLVER_RTR = (
    rtr,
    {
        "radius_max": 1.0,
        "radius_start": 0.1,
        "quality_step_thresh": 0.15,
        "h": IDENT_SYMM_OPER,  # do not change
        "radius_eps": 1e-6,
        "quality_eps": 1e-6,
        # "default_retr_damp": 0.9,
    },
)
PENALTY_START = 0.1
PENALTY_GROWTH = 1.1
INEQ_MULT_START = 0.0
INEQ_MULTS_MIN = -torch.inf
INEQ_MULTS_MAX = torch.inf
EQ_MULT_START = 0.0
EQ_MULTS_MIN = -torch.inf
EQ_MULTS_MAX = torch.inf
SUBSOLVER_TOL_START = 1e-1
SUBSOLVER_TOL_MIN = 1e-3
SUBSOLVER_TOL_DECAY = 0.9
SUBSOLVER_MAX_ITERS = 200
RATIO = 0.8


@dataclass
class OptimResult:
    success: bool
    num_iters: int
    f_hist: torch.Tensor
    ineqs_hist: torch.Tensor
    eqs_hist: torch.Tensor
    p_hist: torch.Tensor


@dataclass
class TestResult:
    start_pos_scaled: np.ndarray
    solver_results: list[OptimResult]


def run_test(
    coord_scaling: float,
    cost_scaling: float,
    constr_scaling: float,
    cost_center: np.ndarray,
    constr_center: np.ndarray,
    constr_radius: float,
    retr: Retraction,
    dist: Distance,
    log_method: LogMapMethod,
    log_covar_method: LogMapCovarMethod,
    start_pos_mean: np.ndarray,
    start_pos_covar: np.ndarray,
    rng_seed: int,
    num_starts: int,
    solver_method: ConstrOptimFn,
    solver_args: dict,
    tol: float,
    max_iters: int,
) -> TestResult:
    M = Manifold[3]

    # define the geometry of the scaled coordinates via the pullback by a linear map
    alpha = 1.0 / coord_scaling
    scaled_metric = MetricLambdaField[M](
        lambda x, y, z: alpha**2
        * coord_repr(
            [
                [1 + 2 * (alpha * x) ** 2, (alpha * x) * (alpha * y), (alpha * x) * (alpha * z)],  # type: ignore
                [(alpha * y) * (alpha * x), 1 + 2 * (alpha * y) ** 2, (alpha * y) * (alpha * z)],  # type: ignore
                [(alpha * z) * (alpha * x), (alpha * z) * (alpha * y), 1 + 2 * (alpha * z) ** 2],  # type: ignore
            ]
        )
    )
    scaled_conn = scaled_metric.levi_civita()

    # define the optimization problem
    f = (
        cost_scaling
        * 0.5
        * RiemSqrDist[M](
            cost_scaling * torch.from_numpy(cost_center),
            scaled_metric,
            log_method,
            log_covar_method,
        )
    )
    g = constr_scaling * (
        RiemSqrDist[M](
            coord_scaling * torch.from_numpy(constr_center),
            scaled_metric,
            log_method,
            log_covar_method,
        )
        - constr_radius**2
    )

    # define the initial conditions
    start_pos_mean_scaled = coord_scaling * start_pos_mean
    start_pos_covar_scaled = coord_scaling**2 * start_pos_covar

    # seeded mutlivariate normal distributions not supported in torch for some reason
    r = np.random.default_rng(rng_seed)
    start_pos_scaled = r.multivariate_normal(
        start_pos_mean_scaled,
        start_pos_covar_scaled,
        num_starts,
    )

    # run solver for each starting position
    solver_results = []
    for i in range(num_starts):
        p0 = start_pos_scaled[i, :]
        print(f"p0: {p0}")
        print(f"initial metric: {scaled_metric(torch.from_numpy(p0)).components}")
        result = solver_method(
            f,
            [g],
            [],
            torch.from_numpy(p0),
            scaled_metric,
            scaled_conn,
            retr,
            dist,
            tol,
            max_iters,
            True,  # save history
            True,  # no debug
            **solver_args,
        )

        if not result.success:
            print("failed!")

        exit()

        # creates a serialiable optimization result
        optim_result = OptimResult(
            result.success,
            result.num_iters,
            result.f_hist,  # type: ignore
            result.ineqs_hist,  # type: ignore
            result.eqs_hist,  # type: ignore
            result.p_hist,  # type: ignore
        )
        solver_results.append(optim_result)

    result = TestResult(
        start_pos_scaled,
        solver_results,
    )
    return result


def run_all_tests(run_dir: Path):

    # generates combinations of methods and scalings along with file names
    configs = []
    for coord_scale in COORD_SCALING_FACTORS:
        configs.extend(
            [
                # rgd
                (
                    f"approx_o1_rgd_{coord_scale}",
                    METHODS_APPROX_O1,
                    SUBSOLVER_RGD,
                    coord_scale,
                ),
                (
                    f"approx_o2_rgd_{coord_scale}",
                    METHODS_APPROX_O2,
                    SUBSOLVER_RGD,
                    coord_scale,
                ),
                (
                    f"approx_o3_rgd_{coord_scale}",
                    METHODS_APPROX_O3,
                    SUBSOLVER_RGD,
                    coord_scale,
                ),
                (
                    f"approx_o4_rgd_{coord_scale}",
                    METHODS_APPROX_O4,
                    SUBSOLVER_RGD,
                    coord_scale,
                ),
                # rtr
                (
                    f"approx_o1_rtr_{coord_scale}",
                    METHODS_APPROX_O1,
                    SUBSOLVER_RTR,
                    coord_scale,
                ),
                (
                    f"approx_o2_rtr_{coord_scale}",
                    METHODS_APPROX_O2,
                    SUBSOLVER_RTR,
                    coord_scale,
                ),
                (
                    f"approx_o3_rtr_{coord_scale}",
                    METHODS_APPROX_O3,
                    SUBSOLVER_RTR,
                    coord_scale,
                ),
                (
                    f"approx_o4_rtr_{coord_scale}",
                    METHODS_APPROX_O4,
                    SUBSOLVER_RTR,
                    coord_scale,
                ),
            ]
        )

    converter = cattrs.Converter()
    for file_name, methods, subsolver, coord_scaling in configs:
        retr, dist, log_method, log_covar_method = methods
        subsolver_method, subsolver_args = subsolver

        print(f"Running {file_name}...")

        # save the configuration under test
        config_subsolver_args = {}
        for key, value in subsolver_args.items():
            config_subsolver_args[key] = str(value)
        config = {
            "coord_scaling": coord_scaling,
            "cost_scaling": COST_SCALING_FACTOR,
            "constr_scaling": CONSTR_SCALING_FACTOR,
            "cost_center": str(COST_CENTER),
            "constr_center": str(CONSTR_CENTER),
            "constr_radius": CONSTR_RADIUS,
            "start_pos_mean": str(START_POS_MEAN),
            "start_pos_covar": str(START_POS_COVAR),
            "rng_seed": RNG_SEED,
            "num_starts": NUM_STARTS,
            "solver_args": {
                "retr": str(retr),
                "dist": str(dist),
                "log_method": str(log_method),
                "log_covar_method": str(log_covar_method),
                # additional args
                "subsolver_method": subsolver_method.__name__,
                "subsolver_args": config_subsolver_args,
                "penalty_start": PENALTY_START,
                "penalty_growth": PENALTY_GROWTH,
                "ineq_mult_start": INEQ_MULT_START,
                "ineq_mults_min": INEQ_MULTS_MIN,
                "ineq_mults_max": INEQ_MULTS_MAX,
                "eq_mult_start": EQ_MULT_START,
                "eq_mults_min": EQ_MULTS_MIN,
                "eq_mults_max": EQ_MULTS_MAX,
                "subsolver_tol_start": SUBSOLVER_TOL_START,
                "subsolver_tol_min": SUBSOLVER_TOL_MIN,
                "subsolver_tol_decay": SUBSOLVER_TOL_DECAY,
                "subsolver_max_iters": SUBSOLVER_MAX_ITERS,
                "ratio": RATIO,
            },
        }
        with open(run_dir / f"{file_name}_config.yaml", "w") as file:
            yaml.safe_dump(config, file)

        # run the test and save the result
        result = run_test(
            coord_scaling,
            COST_SCALING_FACTOR,
            CONSTR_SCALING_FACTOR,
            COST_CENTER,
            CONSTR_CENTER,
            CONSTR_RADIUS,
            retr,
            dist,
            log_method,
            log_covar_method,
            START_POS_MEAN,
            START_POS_COVAR,
            RNG_SEED,
            NUM_STARTS,
            ralm,
            {
                "subsolver_method": subsolver_method,
                "subsolver_args": subsolver_args,
                "penalty_start": PENALTY_START,
                "penalty_growth": PENALTY_GROWTH,
                "ineq_mult_start": INEQ_MULT_START,
                "ineq_mults_min": INEQ_MULTS_MIN,
                "ineq_mults_max": INEQ_MULTS_MAX,
                "eq_mult_start": EQ_MULT_START,
                "eq_mults_min": EQ_MULTS_MIN,
                "eq_mults_max": EQ_MULTS_MAX,
                "subsolver_tol_start": SUBSOLVER_TOL_START,
                "subsolver_tol_min": SUBSOLVER_TOL_MIN,
                "subsolver_tol_decay": SUBSOLVER_TOL_DECAY,
                "subsolver_max_iters": SUBSOLVER_MAX_ITERS,
                "ratio": RATIO,
            },
            TOL,
            MAX_ITERS,
        )
        torch.save(converter.unstructure(result), run_dir / f"{file_name}.pt")


if __name__ == "__main__":
    root_dir = rootutils.setup_root(search_from=__file__, indicator=".project-root")
    results_dir = root_dir / "approx_optim/results"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / f"run_{timestamp}"
    run_dir.mkdir()

    run_all_tests(run_dir)
