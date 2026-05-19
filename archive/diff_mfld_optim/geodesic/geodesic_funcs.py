from enum import Enum

import torch
import numpy as np
from torch.autograd.function import Function

from diff_mfld_optim.geodesic.approx_geod_so import (
    _solve_geod_pos_so,
    _solve_initial_geod_vel_so,
)
from diff_mfld_optim.geometry.metric import MetricField, LeviCivitaConnection
from diff_mfld_optim.geometry.connection import Connection

# wrapper functions for approximate methods


def _exp_map_fo_approx(p, v, conn_coeffs):
    return p + v  # Euclidean


def _log_map_fo_approx(p, q, conn_coeffs):
    return q - p  # Euclidean


def _exp_map_so_approx(p, v, conn_coeffs):
    return _solve_geod_pos_so(p, v, 0.0, 1.0, conn_coeffs)


def _log_map_so_approx(p, q, conn_coeffs):
    return _solve_initial_geod_vel_so(p, q, 0.0, 1.0, conn_coeffs)


# wrapper function to handle torch tensor type conversion to numpy to be used
# with the above approximate methods


# hack! open issue: https://github.com/pytorch/pytorch/issues/91810
# NOTE: this just motivates not using torch as an autograd tool for evaluating
# the function differentials in the future (especially when writing rust lib.)
def _recursive_unwrap_tensor(tensor):
    tensor = torch._C._functorch.get_unwrapped(tensor)
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        return _recursive_unwrap_tensor(tensor)
    return tensor


def detach_numpy(tensor):
    tensor = tensor.detach().cpu()
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        # the shenanigans necessary here are annoying but some cases require
        # this operation to be performed twice
        unwrapped_tensor = _recursive_unwrap_tensor(tensor)

        # deviating from the solution on the open issue for some reason the
        # following can return a larger list (this could possibly be a
        # security issue) so we just clip it to the length of the tensor
        total_len = np.prod(unwrapped_tensor.shape)
        raw_data = np.array(unwrapped_tensor.storage().tolist()[:total_len])
        return np.array(raw_data).reshape(unwrapped_tensor.shape)
    return tensor.numpy()


class _ExpMapWrapper:
    def __init__(self, exp_map):
        self._exp_map = exp_map

    def __call__(self, p: torch.tensor, v: torch.tensor, conn: Connection):
        conn_coeffs = conn(p)

        q = self._exp_map(detach_numpy(p), detach_numpy(v), detach_numpy(conn_coeffs))
        return torch.tensor(q, dtype=p.dtype)


class _LogMapWrapper:
    def __init__(self, log_map):
        self._log_map = log_map

    def __call__(self, p: torch.tensor, q: torch.tensor, conn: Connection):
        conn_coeffs = conn(p)

        v = self._log_map(detach_numpy(p), detach_numpy(q), detach_numpy(conn_coeffs))
        return torch.tensor(v, dtype=p.dtype)


class ExpMethod(Enum):
    APPROX_FO = _ExpMapWrapper(_exp_map_fo_approx)
    APPROX_SO = _ExpMapWrapper(_exp_map_so_approx)

    def __call__(self, p: torch.tensor, v: torch.tensor, conn: Connection):
        return self.value(p, v, conn)


class LogMethod(Enum):
    APPROX_FO = _LogMapWrapper(_log_map_fo_approx)
    APPROX_SO = _LogMapWrapper(_log_map_so_approx)

    def __call__(self, p: torch.tensor, q: torch.tensor, conn: Connection):
        return self.value(p, q, conn)


# usable maps


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

    # @staticmethod
    # def setup_context(ctx, inputs, output):
    #     # all work done in the forward pass
    #     pass

    # @staticmethod
    # def forward(
    #     ctx,
    #     p: torch.Tensor,
    #     q: torch.Tensor,
    #     metric: MetricField,
    #     conn: Connection = None,
    #     log_method=LogMethod.APPROX_SO,
    # ):
    #     print(f"p_requires_grad={p.requires_grad}")
    #     print(f"q_requires_grad={q.requires_grad}")

    #     g = metric(p)  # metric at point p
    #     v = log_method(p, q, conn)  # tangent space at p

    #     print(f"v: {v.requires_grad}")

    #     dv = g.flat(v)  # cotangent space at p (differential)
    #     diff_dist_sqr = -2 * dv

    #     diff_dist_sqr.requires_grad_()

    #     # diff_dist_sqr.requires_grad=True

    #     ctx.save_for_backward(diff_dist_sqr)

    #     dist_sqr = g(v, v) ** 2

    #     print(f"dist_sqr: {dist_sqr.requires_grad}")

    #     dist_sqr.requires_grad_()
    #     print(f"dist_sqr: {dist_sqr.requires_grad}")

    #     return dist_sqr + 0 * p.sum()

    # @staticmethod
    # def backward(ctx, grad_output):
    #     # dv is already the differential of the distance
    #     (diff_dist_sqr,) = ctx.saved_tensors
    #     return (
    #         grad_output * diff_dist_sqr,
    #         # torch needs a "gradient" for each input but these are just
    #         # various parameters so we return None (non-differentiable)
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #     )


def dist_squared_map(
    p: torch.Tensor,
    q: torch.Tensor,
    metric: MetricField,
    conn: Connection = None,
    log_method=LogMethod.APPROX_SO,
) -> torch.tensor:
    # if a custom connection is not defined then use the levi-civita
    # connection derived from the metric
    if conn is None:
        conn = metric.christoffels()
    return DistSquaredMap.apply(p, q, metric, conn, log_method)
