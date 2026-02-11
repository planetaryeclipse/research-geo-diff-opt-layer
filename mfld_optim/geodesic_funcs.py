from enum import Enum

# import torch
from torch.autograd.function import Function

from approx_geod_so import _exp_map_so_approx, _log_map_so_approx
from metric import MetricField, LeviCivitaConnection
from connection import Connection

import numpy as np
from jax import custom_vjp, custom_jvp
from functools import partial
import jax.numpy as jnp

from typing import Tuple


def _exp_map_fo_approx(p, v, conn):
    return p + v  # Euclidean


def _log_map_fo_approx(p, q, conn):
    return q - p  # Euclidean


class ExpMethod(Enum):
    APPROX_FO = _exp_map_fo_approx
    APPROX_SO = _exp_map_so_approx

    def __call__(self, *args):
        self.value(*args)


class LogMethod(Enum):
    APPROX_FO = _log_map_fo_approx
    APPROX_SO = _log_map_so_approx

    def __call__(self, *args):
        self.value(*args)


def exp_map(p, v, conn, method=ExpMethod.APPROX_SO):
    return method(p, v, conn)


def log_map(p, q, conn, method=LogMethod.APPROX_SO):
    return method(p, q, conn)


def dist_map(
    p, q, metric: MetricField, conn: Connection = None, log_method=LogMethod.APPROX_SO
):
    # if connection not defined then use the levi-civita connection from the metric
    if conn is None:
        conn = metric.christoffels()

    # convenient method to compute this value
    v = log_method(p, q, conn)
    return metric(p)(v, v)


@partial(custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def dist_squared_map(
    p, q, metric: MetricField, conn: Connection = None, log_method=LogMethod.APPROX_SO
) -> jnp.ndarray:
    # if connection not defined then use the levi-civita connection from the
    # defined metric field evaluated at p (location where evaluating log map)

    if conn is None:
        conn = metric.christoffels()

    g = metric(p)  # metric at point p
    v = log_method(p, q, conn)  # tangent space at p

    dist = g(v, v) ** 2
    return dist


def _dist_squared_map_fwd(
    p,
    q,
    metric: MetricField,
    conn: Connection,
    log_method: LogMethod,
):
    dist = dist_squared_map(p, q, metric, conn, log_method)

    v = log_method(p, q, conn)
    dv = metric(p).flat(v)  # dev in cotangent space at p
    diff_dist_sqr = -2 * dv

    res = (diff_dist_sqr,)  # can't store non-differentiable residuals here

    return dist, res


def _dist_squared_map_bwd(
    q, metric: MetricField, conn: Connection, log_method: LogMethod, res, g
):
    diff_dist_sqr = res[0]
    return (diff_dist_sqr * g,)


dist_squared_map.defvjp(_dist_squared_map_fwd, _dist_squared_map_bwd)

# @dist_squared_map.defjvp
# def _dist_squared_map_jvp(
#     q,
#     metric: MetricField,
#     conn: Connection,
#     log_method,
#     # diff arguments
#     primals,
#     tangents,
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     (p,) = primals
#     (p_dot,) = tangents

#     if conn is None:
#         conn = metric.christoffels()

#     g = metric(p)  # metric at point p
#     v = log_method(p, q, conn)  # tangent space at p
#     dist_sqr = g(v, v) ** 2

#     # NOTE: JAX considers the tangents to be the gradient which works given
#     # the assumption of Euclidean space but this breaks down for manifolds
#     # given the tangent and cotangent space are related by the metric. Thereby
#     # we consider JAX's tangents to be the cotangents and thereby our notion
#     # of chain rule differentation holds correctly

#     dv = g.flat(v)  # cotangent space at p (differential
#     diff_dist_sqr = 2 * g.inv(p_dot, p_dot)

#     print(f"diff_dist_sqr: {diff_dist_sqr}")

#     primal_out = dist_sqr
#     tangent_out = diff_dist_sqr  #  * p_dot

#     print(f"primal out: {primal_out}")
#     print(f"tangent out: {tangent_out}")

#     return primal_out, tangent_out


# class DistSquaredMap(Function):
#     # by defining the distance map as a torch function then we can evaluate the
#     # differential (cotangent space) of the function using the torch autograd
#     # system (we consider torch gradient to be the differential here as torch
#     # was not built for differential geometry) automatically and can therefore
#     # easily define the cost/constraint functions for an optimization problem
#     # NOTE: do not use this as part of a training pipeline

#     @staticmethod
#     def forward(
#         ctx,
#         p,
#         q,
#         metric: MetricField,
#         conn: Connection = None,
#         log_method=LogMethod.APPROX_SO,
#     ):
#         # if a custom connection is not defined then use the levi-civita
#         # connection derived from the metric
#         if conn is None:
#             conn = metric.christoffels()

#         g = metric(p)  # metric at point p

#         v = log_method(p, q, conn)  # tangent space at p
#         dist_sqr = g(v, v) ** 2

#         dv = g.flat(v)  # cotangent space at p (differential)
#         diff_dist_sqr = -2 * dv
#         ctx.save_for_backward(diff_dist_sqr)

#         return dist_sqr

#     @staticmethod
#     def backward(ctx, grad_output):
#         # dv is already the differential of the distance
#         (diff_dist_sqr,) = ctx.saved_tensors
#         return (
#             grad_output * diff_dist_sqr,
#             # torch needs a "gradient" for each input but these are just
#             # various parameters so we return None (non-differentiable)
#             None,
#             None,
#             None,
#             None,
#             None,
#         )
