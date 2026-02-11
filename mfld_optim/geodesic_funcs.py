from enum import Enum

import torch
from torch.autograd.function import Function

from approx_geod_so import _exp_map_so_approx, _log_map_so_approx
from metric import MetricField, LeviCivitaConnection
from connection import Connection


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


class DistSquaredMap(Function):
    # by defining the distance map as a torch function then we can evaluate the
    # differential (cotangent space) of the function using the torch autograd
    # system (we consider torch gradient to be the differential here as torch
    # was not built for differential geometry) automatically and can therefore
    # easily define the cost/constraint functions for an optimization problem
    # NOTE: do not use this as part of a training pipeline

    generate_vmap_rule = True

    @staticmethod
    def forward(
        p,
        q,
        metric: MetricField,
        conn: Connection = None,
        log_method=LogMethod.APPROX_SO,
    ):
        # if a custom connection is not defined then use the levi-civita
        # connection derived from the metric
        if conn is None:
            conn = metric.christoffels()

        g = metric(p)  # metric at point p

        v = log_method(p, q, conn)  # tangent space at p
        dist_sqr = g(v, v) ** 2

        return dist_sqr

    @staticmethod
    def setup_context(ctx, inputs, output):
        (p, q, metric, conn, log_method) = inputs

        if conn is None:
            conn = metric.christoffels()

        g = metric(p)
        v = log_method(p, q, conn)  # tangent space at p

        dv = g.flat(v)  # cotangent space at p (differential)
        diff_dist_sqr = -2 * dv
        ctx.save_for_backward(diff_dist_sqr)

    @staticmethod
    def backward(ctx, grad_output):
        # dv is already the differential of the distance
        (diff_dist_sqr,) = ctx.saved_tensors
        return (
            grad_output * diff_dist_sqr,
            # torch needs a "gradient" for each input but these are just
            # various parameters so we return None (non-differentiable)
            None,
            None,
            None,
            None,
            None,
        )
