import itertools
import numpy as np
import torch

from enum import Enum
from scipy.optimize import root

from diff_mfld_optim.geometry.metric import RnMetricField
from diff_mfld_optim.geometry.connection import Connection

from typing import Tuple


def _inspect_conn_coeffs(conn_coeffs: np.ndarray) -> Tuple[int, int]:
    n = conn_coeffs.shape[0]
    r = conn_coeffs.shape[2]

    if n != conn_coeffs.shape[1]:
        raise ValueError(
            f"connection coefficients must share length on dims 0, 1: dim0={n}, dim1={conn_coeffs.shape[1]}"
        )

    return n, r


def _solve_geod_pos_so(
    p: np.ndarray, v: np.ndarray, t0: float, t: float, conn_coeffs: np.ndarray
) -> np.ndarray:
    # solves for the updated position along the geodesic using the second order
    # approximation of the geodesic (requires a tangent bundle connection)

    n, r = _inspect_conn_coeffs(conn_coeffs)
    if n != r:
        raise ValueError(
            f"connection must be defined on tangent bundle so n == r: n={n}, {r}"
        )

    q = np.zeros((n), dtype=p.dtype)  # otherwise causes issues in the chain
    for k in range(n):
        q[k] += p[k] + v[k] * (t - t0)
        for i, j in itertools.product(range(n), range(n)):
            q[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2

    return q


def _initial_geod_vel_f_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
    n = len(v)

    f = np.zeros(n, dtype=p.dtype)
    for k in range(n):
        f[k] = v[k] * (t - t0) + (p[k] - q[k])
        for i, j in itertools.product(range(n), range(n)):
            f[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2
    return f


def _initial_geod_vel_fprime_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
    n = len(v)

    df_dv = np.zeros((n, n), dtype=p.dtype)
    for k, i in itertools.product(range(n), range(n)):
        if k == i:
            df_dv[k] += t - t0
        for j in range(n):
            df_dv[k, i] += (
                -0.5
                * (conn_coeffs[k, i, j] + conn_coeffs[k, j, i])
                * v[j]
                * (t - t0) ** 2
            )
    return df_dv


def _solve_initial_geod_vel_so(
    p: np.ndarray, q: np.ndarray, t0: float, t: float, conn_coeffs: np.ndarray
) -> np.ndarray:
    # solves for the initial velocity along the geodesic using the second order
    # approximation of the geodesic (requires a tangent bundle connection)

    n, r = _inspect_conn_coeffs(conn_coeffs)
    if n != r:
        raise ValueError(
            f"connection must be defined on tangent bundle so n == r: n={n}, {r}"
        )

    v_guess = (q - p) / (t - t0)  # Euclidean is the guess

    result = root(
        _initial_geod_vel_f_so,
        v_guess,
        args=(p, q, conn_coeffs, t, t0),
        jac=_initial_geod_vel_fprime_so,
    )

    if result.success:
        # root operation computes a float64 for some reason
        return result.x.astype(dtype=p.dtype)
    else:
        print(f"p: {p}")
        print(f"q: {q}")
        print(f"t0: {t0}")
        print(f"t: {t}")

        raise ValueError(f"failed to find solution to logarithmic map: {result}")
