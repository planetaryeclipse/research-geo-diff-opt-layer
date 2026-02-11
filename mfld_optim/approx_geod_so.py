import itertools

# import numpy as np
# import torch

from enum import Enum
from aenum import member
from scipy.optimize import root

from metric import RnMetricField
from connection import Connection

import numpy as np
import jax.numpy as jnp


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
        f[k] = v[k] * (t - t0) + (p[k] - q[k])
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
        return result.x
    else:
        raise ValueError(
            f"Failed to find solution to logarithmic map: {result.message}"
        )


# def _solve_initial_geod_vel_so(
#     p, q, t0: float, t: float, conn: Connection, eps=1e-5, max_iters=100
# ) -> np.ndarray:
#     # solves for the initial velocity along the geodesic using the second order
#     # approximation of the geodesic (requires a tangent bundle connection)

#     if conn.n != conn.r:
#         raise ValueError("tangent bundle is a n-dim vector bundle")

#     conn_coeffs: np.ndarray = conn(p)
#     v_guess = (q - p) / (t - t0)  # Euclidean is the guess

#     # custom root finding as this needs to be compatible with jax jit

#     prev_v = v_guess
#     v = v_guess

#     for _ in range(max_iters):
#         f = _initial_geod_vel_f_so(v, p, q, conn_coeffs, t, t0)
#         jac = _initial_geod_vel_fprime_so(v, p, q, conn_coeffs, t, t0)

#         v -= jnp.linalg.inv(jac) @ f
#         if (prev_v - v).norm() <= eps:
#             return v

#     raise ValueError(f"Failed to find solution to logarithmic map in {max_iters} iters")


# the above internal types handle computation strictly as numpy types so these
# wrapper methods need to ensure the correct input and outputs


def _exp_map_so_approx(p: jnp.ndarray, v: jnp.ndarray, conn: Connection) -> jnp.ndarray:
    p, v = np.asarray(p), np.asarray(v)  # internal helpers use numpy
    return jnp.asarray(_solve_geod_pos_so(p, v, 0.0, 1.0, conn))


def _log_map_so_approx(p: jnp.ndarray, q: jnp.ndarray, conn) -> jnp.ndarray:
    p, q = np.asarray(p), np.asarray(q)  # internal helpers use numpy
    return jnp.asarray(_solve_initial_geod_vel_so(p, q, 0.0, 1.0, conn))


def test():
    p = p = jnp.array([1.0, 2.0])
    q = jnp.array([4.0, -1.0])

    g = RnMetricField(2)
    conn = g.christoffels()

    print(_log_map_so_approx(p, q, conn))


if __name__ == "__main__":
    test()
