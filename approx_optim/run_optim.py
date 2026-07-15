from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
import cattrs
import rootutils
import torch
import numpy as np


from dmol.diff_mfld.connection.methods.geod_log_diff import LogMapCovarMethod
from dmol.diff_mfld.connection.methods.methods import (
    Distance,
    DistanceMethod,
    ExpMapMethod,
    LogMapMethod,
)
from dmol.diff_mfld.field.riem_fields import RiemSqrDist
from dmol.diff_mfld.field.util import coord_repr
from dmol.diff_mfld.mfld import Manifold
from dmol.diff_mfld.riemann import MetricLambdaField
from dmol.optim.constr.ralm import ralm
from dmol.optim.methods import ConstrOptimFn, Retraction
from dmol.optim.unconstr.rgd import rgd
from dmol.optim.unconstr.rtr import rtr


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
    # solver args
    tol = 1e-3
    max_iters = 200
    methods_approx_o1 = (
        ExpMapMethod.APPROX_O1,
        DistanceMethod.APPROX_O1,
        LogMapMethod.APPROX_O1,
        LogMapCovarMethod.APPROX_O1,
    )
    methods_approx_o2 = (
        ExpMapMethod.APPROX_O2,
        DistanceMethod.APPROX_O2,
        LogMapMethod.APPROX_O2,
        LogMapCovarMethod.APPROX_O2,
    )
    methods_approx_o3 = (
        ExpMapMethod.APPROX_O3,
        DistanceMethod.APPROX_O3,
        LogMapMethod.APPROX_O3,
        LogMapCovarMethod.APPROX_O3,
    )
    methods_approx_o4 = (
        ExpMapMethod.APPROX_O4,
        DistanceMethod.APPROX_O4,
        LogMapMethod.APPROX_O4,
        LogMapCovarMethod.APPROX_O4,
    )

    # problem setup
    coord_scaling_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.50]
    cost_scaling_factor = 0.02
    constr_scaling_factor = 0.02
    cost_center = np.array([-5.0, -1.0, 3.0])
    constr_center = np.array([-2.0, 3.0, 1.0])
    constr_radius = 3.0
    start_pos_mean = np.array([2.0, -5.0, -8.0])
    start_pos_covar = 2.0**2 * np.eye(3)
    num_starts = 20
    rng_seed = 42

    # ralm specific arguments
    subsolver_rgd = (rgd, {"damp": 0.15})
    ident_symm_oper = lambda p: p
    subsolver_rtr = (
        rtr,
        {
            "radius_max": 1.0,
            "radius_start": 0.1,
            "quality_step_thresh": 0.15,
            "h": ident_symm_oper,  # do not change
            "radius_eps": 1e-6,
            "quality_eps": 1e-6,
            # "default_retr_damp": 0.9,
        },
    )
    penalty_start = 0.1
    penalty_growth = 1.1
    ineq_mult_start = 0.0
    ineq_mults_min = -torch.inf
    ineq_mults_max = torch.inf
    eq_mult_start = 0.0
    eq_mults_min = -torch.inf
    eq_mults_max = torch.inf
    subsolver_tol_start = 1e-1
    subsolver_tol_min = 1e-3
    subsolver_tol_decay = 0.9
    subsolver_max_iters = 200
    ratio = 0.8

    # generates configs along with file prefixes
    configs = []
    for coord_scale in coord_scaling_factors:
        configs.extend(
            [
                # rgd
                (
                    f"approx_o1_rgd_{coord_scale}",
                    methods_approx_o1,
                    subsolver_rgd,
                    coord_scale,
                ),
                (
                    f"approx_o2_rgd_{coord_scale}",
                    methods_approx_o2,
                    subsolver_rgd,
                    coord_scale,
                ),
                (
                    f"approx_o3_rgd_{coord_scale}",
                    methods_approx_o3,
                    subsolver_rgd,
                    coord_scale,
                ),
                (
                    f"approx_o4_rgd_{coord_scale}",
                    methods_approx_o4,
                    subsolver_rgd,
                    coord_scale,
                ),
                # rtr
                (
                    f"approx_o1_rtr_{coord_scale}",
                    methods_approx_o1,
                    subsolver_rtr,
                    coord_scale,
                ),
                (
                    f"approx_o2_rtr_{coord_scale}",
                    methods_approx_o2,
                    subsolver_rtr,
                    coord_scale,
                ),
                (
                    f"approx_o3_rtr_{coord_scale}",
                    methods_approx_o3,
                    subsolver_rtr,
                    coord_scale,
                ),
                (
                    f"approx_o4_rtr_{coord_scale}",
                    methods_approx_o4,
                    subsolver_rtr,
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
            "cost_scaling": cost_scaling_factor,
            "constr_scaling": constr_scaling_factor,
            "cost_center": str(cost_center),
            "constr_center": str(constr_center),
            "constr_radius": constr_radius,
            "start_pos_mean": str(start_pos_mean),
            "start_pos_covar": str(start_pos_covar),
            "rng_seed": rng_seed,
            "num_starts": num_starts,
            "solver_args": {
                "retr": str(retr),
                "dist": str(dist),
                "log_method": str(log_method),
                "log_covar_method": str(log_covar_method),
                # additional args
                "subsolver_method": subsolver_method.__name__,
                "subsolver_args": config_subsolver_args,
                "penalty_start": penalty_start,
                "penalty_growth": penalty_growth,
                "ineq_mult_start": ineq_mult_start,
                "ineq_mults_min": ineq_mults_min,
                "ineq_mults_max": ineq_mults_max,
                "eq_mult_start": eq_mult_start,
                "eq_mults_min": eq_mults_min,
                "eq_mults_max": eq_mults_max,
                "subsolver_tol_start": subsolver_tol_start,
                "subsolver_tol_min": subsolver_tol_min,
                "subsolver_tol_decay": subsolver_tol_decay,
                "subsolver_max_iters": subsolver_max_iters,
                "ratio": ratio,
            },
        }
        with open(run_dir / f"{file_name}_config.yaml", "w") as file:
            yaml.safe_dump(config, file)

        # run the test and save the result
        result = run_test(
            coord_scaling,
            cost_scaling_factor,
            constr_scaling_factor,
            cost_center,
            constr_center,
            constr_radius,
            retr,
            dist,
            log_method,
            log_covar_method,
            start_pos_mean,
            start_pos_covar,
            rng_seed,
            num_starts,
            ralm,
            {
                "subsolver_method": subsolver_method,
                "subsolver_args": subsolver_args,
                "penalty_start": penalty_start,
                "penalty_growth": penalty_growth,
                "ineq_mult_start": ineq_mult_start,
                "ineq_mults_min": ineq_mults_min,
                "ineq_mults_max": ineq_mults_max,
                "eq_mult_start": eq_mult_start,
                "eq_mults_min": eq_mults_min,
                "eq_mults_max": eq_mults_max,
                "subsolver_tol_start": subsolver_tol_start,
                "subsolver_tol_min": subsolver_tol_min,
                "subsolver_tol_decay": subsolver_tol_decay,
                "subsolver_max_iters": subsolver_max_iters,
                "ratio": ratio,
            },
            tol,
            max_iters,
        )
        torch.save(converter.unstructure(result), run_dir / f"{file_name}.pt")


if __name__ == "__main__":
    root_dir = rootutils.setup_root(search_from=__file__)
    results_dir = root_dir / "approx_optim/results"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = results_dir / f"run_{timestamp}"
    run_dir.mkdir()

    run_all_tests(run_dir)
