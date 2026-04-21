import numpy as np
import torch

import warnings
import inspect

from scipy.optimize import root
from src.diff_mfld.geometry.connection import Connection

from typing import List, Callable


# wrapper class which matches the expected signature and then computes all required connection coefficients (and
# partials if applicable) before passing it to the approximate methods below

# NOTE: cannot specify callable arguments as [np.ndarray, np.ndarray, ...] so I just use the full variadic below, it is
# assumed that there are always 2 occupied ndarray arguments and the remainder are the connection coefficients with
# increasing derivative order of partials


def _count_conn_coeff_args(map):
    # measures the number of parameters after the alpha parameter (correspond to the connection
    # coefficients and any remaining partials after that)
    sig = inspect.signature(map)
    coeffs_count = len(sig.parameters) - 3

    return coeffs_count


def _compute_conn_coeff_args(
        count: int, p: torch.Tensor, conn: Connection
) -> List[np.ndarray]:
    conn_coeffs_with_partials = []
    if count > 0:
        conn_coeffs_with_partials.append(conn(p).detach().numpy())
    for deriv_order in range(1, count):
        conn_coeffs_with_partials.append(conn.partials(p, deriv_order).detach().numpy())

    return conn_coeffs_with_partials


class ApproxExpMapWrapper:
    def __init__(self, exp_map: Callable[..., np.ndarray]):
        self._exp_map = exp_map
        self._coeffs_count = _count_conn_coeff_args(exp_map)

    def __call__(
            self, p: torch.Tensor, v: torch.Tensor, conn: Connection, alpha: float
    ) -> torch.Tensor:
        conn_coeffs_with_partials = _compute_conn_coeff_args(
            self._coeffs_count, p, conn
        )

        p_numpy = p.detach().numpy()
        v_numpy = v.detach().numpy()

        q_numpy = self._exp_map(p_numpy, v_numpy, alpha, *conn_coeffs_with_partials)
        return torch.tensor(q_numpy, dtype=p.dtype)


class ApproxLogMapWrapper:
    def __init__(self, log_map: Callable[..., np.ndarray]):
        self._log_map = log_map
        self._coeffs_count = _count_conn_coeff_args(log_map)

    def __call__(
            self, p: torch.Tensor, q: torch.Tensor, conn: Connection, alpha: float
    ) -> torch.Tensor:
        conn_coeffs_with_partials = _compute_conn_coeff_args(
            self._coeffs_count, p, conn)

        p_numpy = p.detach().numpy()
        q_numpy = q.detach().numpy()

        # print(f"len: {len(conn_coeffs_with_partials)}")

        v_numpy = self._log_map(p_numpy, q_numpy, alpha, *conn_coeffs_with_partials)
        return torch.tensor(v_numpy, dtype=p.dtype)


# approximated exponential and logarithmic maps

# NOTE: the following can technically be extended to computing values along the geodesic in a generic manner but we're
# explicitly handling the exponential adn logarithmic cases where t=1 so we just omit the (t-0)^k term in the Taylow
# expansions in the functions below


def approx_exp_map_o1(p: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    return p + f0(v) * alpha


def approx_exp_map_o2(
        p: np.ndarray, v: np.ndarray, alpha: float, conn_coeffs: np.ndarray
) -> np.ndarray:
    return p + f0(v) * alpha + 1. / 2. * f1(v, conn_coeffs) * alpha**2


def approx_exp_map_o3(
        p: np.ndarray,
        v: np.ndarray,
        alpha: float,
        conn_coeffs: np.ndarray,
        conn_coeffs_partials: np.ndarray,
) -> np.ndarray:
    return p + f0(v) * alpha + 1. / 2. * f1(v, conn_coeffs) * alpha**2 + 1. / 6. * f2(v, conn_coeffs, conn_coeffs_partials) * alpha**3


def _solve_approx_log_map(f_fn, fprime_fn, p, q, order, alpha):
    v_guess = approx_log_map_o1(p, q, alpha)  # uses initial Euclidean gues
    result = root(f_fn, v_guess, jac=fprime_fn)

    n = p.shape[0]
    if result.success:
        v = result.x[:n]
    else:
        print(
            f"failed to find solution to order {order} approx log map, falling back to order 1 estimate"
        )
        v = v_guess
    return v.astype(dtype=p.dtype)  # always returns float64 for some reason


def approx_log_map_o1(p: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    return (q - p) / alpha  # no need to use solver here


def approx_log_map_o2(
        p: np.ndarray, q: np.ndarray, alpha: float, conn_coeffs: np.ndarray
) -> np.ndarray:
    f_fn = lambda v: -q + (p + f0(v) * alpha + 1. / 2. * f1(v, conn_coeffs) * alpha**2)
    fprime_fn = lambda v: f0_jacob(v) * alpha + 1. / 2. * f1_jacob(v, conn_coeffs) * alpha**2

    return _solve_approx_log_map(f_fn, fprime_fn, p, q, 2, alpha)


def approx_log_map_o3(
        p: np.ndarray,
        q: np.ndarray,
        alpha: float,
        conn_coeffs: np.ndarray,
        conn_coeffs_fo_partials: np.ndarray,
) -> np.ndarray:
    f_fn = lambda v: -q + (
            p + f0(v) * alpha + 1. / 2. * f1(v, conn_coeffs) * alpha**2 + 1. / 6. * f2(v, conn_coeffs, conn_coeffs_fo_partials) * alpha**3
    )
    fprime_fn = (
        lambda v: f0_jacob(v) * alpha
                  + 1. / 2. * f1_jacob(v, conn_coeffs) * alpha**2
                  + 1. / 6. * f2_jacob(v, conn_coeffs, conn_coeffs_fo_partials) * alpha**3
    )

    return _solve_approx_log_map(f_fn, fprime_fn, p, q, 3, alpha)


# implements the higher order component calculations


def f0(y: np.ndarray) -> np.ndarray:
    return y


def f0_jacob(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    return np.eye(n)


def f1(y: np.ndarray, conn_coeffs: np.ndarray) -> np.ndarray:
    result = -np.einsum("kab,a,b->k", conn_coeffs, y, y)

    return result


def f1_jacob(y: np.ndarray, conn_coeffs: np.ndarray) -> np.ndarray:
    result = -np.einsum("kab,b->ka", conn_coeffs, y)
    result += -np.einsum("kab,a->kb", conn_coeffs, y)

    return result


def f2(
        y: np.ndarray, conn_coeffs: np.ndarray, conn_coeffs_fo_partials: np.ndarray
) -> np.ndarray:
    result = -np.einsum("kabc,a,b,c->k", conn_coeffs_fo_partials, y, y, y)
    result += np.einsum("kab,a,bcd,c,d->k", conn_coeffs, y, conn_coeffs, y, y)
    result += np.einsum("kab,acd,c,d,b->k", conn_coeffs, conn_coeffs, y, y, y)

    return result


def f2_jacob(
        y: np.ndarray, conn_coeffs: np.ndarray, conn_coeffs_partials: np.ndarray
) -> np.ndarray:
    # first term
    result = -np.einsum("kabc,b,c->ka", conn_coeffs_partials, y, y)
    result += -np.einsum("kabc,a,c->kb", conn_coeffs_partials, y, y)
    result += -np.einsum("kabc,a,b->kc", conn_coeffs_partials, y, y)

    # second term
    result += np.einsum("kab,bcd,c,d->ka", conn_coeffs, conn_coeffs, y, y)
    result += np.einsum("kab,a,bcd,d->kc", conn_coeffs, y, conn_coeffs, y)
    result += np.einsum("kab,a,bcd,c->kd", conn_coeffs, y, conn_coeffs, y)

    # third term
    result += np.einsum("kab,acd,c,d->kb", conn_coeffs, conn_coeffs, y, y)
    result += np.einsum("kab,acd,b,d->kc", conn_coeffs, conn_coeffs, y, y)
    result += np.einsum("kab,acd,b,c->kd", conn_coeffs, conn_coeffs, y, y)

    return result

# older approximate code from research paper (now unused)

# def _inspect_conn_coeffs(conn_coeffs: np.ndarray) -> Tuple[int, int]:
#     n = conn_coeffs.shape[0]
#     r = conn_coeffs.shape[2]

#     if n != conn_coeffs.shape[1]:
#         raise ValueError(
#             f"connection coefficients must share length on dims 0, 1: dim0={n}, dim1={conn_coeffs.shape[1]}"
#         )

#     return n, r


# def _solve_geod_pos_so(
#     p: np.ndarray, v: np.ndarray, t0: float, t: float, conn_coeffs: np.ndarray
# ) -> np.ndarray:
#     # solves for the updated position along the geodesic using the second order
#     # approximation of the geodesic (requires a tangent bundle connection)

#     n, r = _inspect_conn_coeffs(conn_coeffs)
#     if n != r:
#         raise ValueError(
#             f"connection must be defined on tangent bundle so n == r: n={n}, {r}"
#         )

#     q = np.zeros((n), dtype=p.dtype)  # otherwise causes issues in the chain
#     for k in range(n):
#         q[k] += p[k] + v[k] * (t - t0)
#         for i, j in itertools.product(range(n), range(n)):
#             q[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2

#     return q


# def _initial_geod_vel_f_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
#     n = len(v)

#     f = np.zeros(n, dtype=p.dtype)
#     for k in range(n):
#         f[k] = v[k] * (t - t0) + (p[k] - q[k])
#         for i, j in itertools.product(range(n), range(n)):
#             f[k] -= 0.5 * conn_coeffs[k, i, j] * v[i] * v[j] * (t - t0) ** 2
#     return f


# def _initial_geod_vel_fprime_so(v, p, q, conn_coeffs, t, t0) -> np.ndarray:
#     n = len(v)

#     df_dv = np.zeros((n, n), dtype=p.dtype)
#     for k, i in itertools.product(range(n), range(n)):
#         if k == i:
#             df_dv[k] += t - t0
#         for j in range(n):
#             df_dv[k, i] += (
#                 -0.5
#                 * (conn_coeffs[k, i, j] + conn_coeffs[k, j, i])
#                 * v[j]
#                 * (t - t0) ** 2
#             )
#     return df_dv


# def _solve_initial_geod_vel_so(
#     p: np.ndarray, q: np.ndarray, t0: float, t: float, conn_coeffs: np.ndarray
# ) -> np.ndarray:
#     # solves for the initial velocity along the geodesic using the second order
#     # approximation of the geodesic (requires a tangent bundle connection)

#     n, r = _inspect_conn_coeffs(conn_coeffs)
#     if n != r:
#         raise ValueError(
#             f"connection must be defined on tangent bundle so n == r: n={n}, {r}"
#         )

#     v_guess = (q - p) / (t - t0)  # Euclidean is the guess

#     result = root(
#         _initial_geod_vel_f_so,
#         v_guess,
#         args=(p, q, conn_coeffs, t, t0),
#         jac=_initial_geod_vel_fprime_so,
#     )

#     if result.success:
#         # root operation computes a float64 for some reason
#         return result.x.astype(dtype=p.dtype)
#     else:
#         warnings.warn(
#             "failed to find solution to logarithmic map given conditions "
#             f"p={p}, q={q}, t0={t0}, t={t}, conn_coeffs={conn_coeffs}, "
#             "falling back to Euclidean estimate"
#         )
#         return v_guess
