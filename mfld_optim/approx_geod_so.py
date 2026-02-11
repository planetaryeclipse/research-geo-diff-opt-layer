import itertools
import numpy as np
import torch

from enum import Enum
from aenum import member
from scipy.optimize import root

from connection import Connection


def _solve_geod_pos_so(p, v, t0: float, t: float, conn: Connection) -> np.ndarray:
    # solves for the updated position along the geodesic using the second order
    # approximation of the geodesic (requires a tangent bundle connection)

    if conn.n != conn.r:
        raise ValueError("tangent bundle is a n-dim vector bundle")

    conn_coeffs: np.ndarray = conn(p)
    q = np.zeros(conn.n)
    for k in range(conn.n):
        q[k] += p[k] + v[k] * (t - t0)
        for i, j in itertools.product(range(conn.n), range(conn.r)):
            q[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2

    return q


def _initial_geod_vel_f_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
    n = len(v)

    f = np.zeros(n)
    for k in range(n):
        f[k] = v[k] * (t - t0) + (q[k] - p[k])
        for i, j in itertools.product(range(n), range(n)):
            f[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2
    return f


def _initial_geod_vel_fprime_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
    n = len(v)

    df_dv = np.zeros((n, n))
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
    p, q, t0: float, t: float, conn: Connection
) -> np.ndarray:
    # solves for the initial velocity along the geodesic using the second order
    # approximation of the geodesic (requires a tangent bundle connection)

    if conn.n != conn.r:
        raise ValueError("tangent bundle is a n-dim vector bundle")

    conn_coeffs: np.ndarray = conn(p)
    v_guess = (q - p) / (t - t0)  # Euclidean is the guess

    result = root(
        _initial_geod_vel_f_so,
        v_guess,
        args=(p, q, conn_coeffs, t, t0),
        jac=_initial_geod_vel_fprime_so,
    )

    if result.success:
        # keep everything as a torch tensor of float32 to prevent shenanigans
        return torch.tensor(result.x, dtype=torch.float32)
    else:
        raise ValueError(
            f"Failed to find solution to logarithmic map: {result.message}"
        )


def _exp_map_so_approx(p, v, conn):
    return _solve_geod_pos_so(p, v, 0.0, 1.0, conn)


def _log_map_so_approx(p, q, conn):
    return _solve_initial_geod_vel_so(p, q, 0.0, 1.0, conn)
