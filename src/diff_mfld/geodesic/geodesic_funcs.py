import torch
import numpy as np

import warnings

from enum import Enum
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import root_scalar, root

from src.diff_mfld.geodesic.approx_geod import (
    approx_exp_map_o1,
    approx_exp_map_o2,
    approx_exp_map_o3,
    approx_log_map_o1,
    approx_log_map_o2,
    approx_log_map_o3,
    ApproxExpMapWrapper,
    ApproxLogMapWrapper
)
from src.diff_mfld.geometry.metric import MetricField, LeviCivitaConnection
from src.diff_mfld.geometry.connection import Connection

LOG_MAP_INITIAL_MESH_SIZE = 10

LOG_MAP_BVP_MAX_NODES = 500
LOG_MAP_BVP_TOL = 1E-2


# exact methods


def _geod_ivp_fn_batched(t, y: np.ndarray, n: int, conn: Connection):
    # nodes = y.shape[1]
    # print(f"Num nodes: {nodes}")

    p, v = y[:n, :], y[n:, :]  # [n, samples], [n, samples]
    conn_coeffs = conn(torch.tensor(p)).detach().numpy()  # [n, n, n, samples]

    dot_p = v
    dot_v = -np.einsum("kijb,ib,jb->kb", conn_coeffs, v, v)

    return np.concat((dot_p, dot_v), axis=0)


def _geod_ivp_fn(t, y: np.ndarray, n: int, conn: Connection):
    p, v = y[:n], y[n:]  # [n], [n]
    conn_coeffs = conn(torch.tensor(p)).detach().numpy()  # [n, n, n]

    dot_p = v
    dot_v = -np.einsum("kij,i,j->k", conn_coeffs, v, v)

    return np.concat((dot_p, dot_v), axis=0)


def ivp_exp_map(p: torch.Tensor, v: torch.Tensor, conn: Connection, alpha: float = 1.0) -> torch.Tensor:
    n = p.shape[0]
    result = solve_ivp(
        _geod_ivp_fn,
        [0.0, alpha],
        np.concat((p.detach().numpy(), v.detach().numpy())),
        method="Radau",  # implicit scheme to improve stability
        args=(n, conn),
    )

    y_f = result.y[:, -1]
    p_f = y_f[:n]
    return torch.tensor(p_f, dtype=p.dtype)


def _geod_bc_fn(ya, yb, p: np.ndarray, q: np.ndarray, n: int):
    pos_a, vel_a = ya[:n], ya[n:]
    pos_b, vel_b = yb[:n], yb[n:]

    # print(f'pos_a: {pos_a}\t\t\t\tvel_a: {vel_a}')
    # print(f'pos_b: {pos_b}\t\t\t\tvel_b: {vel_b}')

    return np.hstack((
        pos_a - p,
        pos_b - q
    ))


def bvp_log_map(p: torch.Tensor, q: torch.Tensor, conn: Connection, alpha: float = 1.0) -> torch.Tensor:
    p_numpy, q_numpy = p.detach().numpy(), q.detach().numpy()

    t_initial_mesh = np.linspace(0.0, alpha, LOG_MAP_INITIAL_MESH_SIZE)

    # constructs Euclidean guess
    p_guess_mesh = np.linspace(p_numpy, q_numpy, LOG_MAP_INITIAL_MESH_SIZE).T
    v_guess_mesh = np.tile(np.reshape(q - p, (len(q), 1)), (1, LOG_MAP_INITIAL_MESH_SIZE))
    y_guess_mesh = np.concat((p_guess_mesh, v_guess_mesh), axis=0)

    n = p.shape[0]
    result = solve_bvp(
        lambda t, y: _geod_ivp_fn_batched(t, y, n, conn),
        lambda ya, yb: _geod_bc_fn(ya, yb, p_numpy, q_numpy, n),
        t_initial_mesh,
        y_guess_mesh,
        tol=LOG_MAP_BVP_TOL,
        max_nodes=LOG_MAP_BVP_MAX_NODES
    )

    if result.success:
        v = result.y[n:, 0]
    else:
        print(f"failed to find solution to bvp log map, falling back to eulidean estimate")
        v = q_numpy - p_numpy
    return torch.tensor(v, dtype=p.dtype)


# wrapper functions for approximate methods


# def _exp_map_fo_approx(p, v, conn_coeffs):
#     return p + v  # Euclidean


# def _log_map_fo_approx(p, q, conn_coeffs):
#     return q - p  # Euclidean


# def _exp_map_so_approx(p, v, conn_coeffs):
#     return _solve_geod_pos_so(p, v, 0.0, 1.0, conn_coeffs)


# def _log_map_so_approx(p, q, conn_coeffs):
#     return _solve_initial_geod_vel_so(p, q, 0.0, 1.0, conn_coeffs)


# # wrapper function to handle torch tensor type conversion to numpy to be used
# # with the above approximate methods


# # hack! open issue: https://github.com/pytorch/pytorch/issues/91810
# # NOTE: this just motivates not using torch as an autograd tool for evaluating
# # the function differentials in the future (especially when writing rust lib.)
# def _recursive_unwrap_tensor(tensor):
#     tensor = torch._C._functorch.get_unwrapped(tensor)
#     if torch._C._functorch.is_gradtrackingtensor(tensor):
#         return _recursive_unwrap_tensor(tensor)
#     return tensor


# def detach_numpy(tensor):
#     tensor = tensor.detach().cpu()
#     if torch._C._functorch.is_gradtrackingtensor(tensor):
#         # the shenanigans necessary here are annoying but some cases require
#         # this operation to be performed twice
#         unwrapped_tensor = _recursive_unwrap_tensor(tensor)

#         # deviating from the solution on the open issue for some reason the
#         # following can return a larger list (this could possibly be a
#         # security issue) so we just clip it to the length of the tensor
#         total_len = np.prod(unwrapped_tensor.shape)
#         raw_data = np.array(unwrapped_tensor.storage().tolist()[:total_len])
#         return np.array(raw_data).reshape(unwrapped_tensor.shape)
#     return tensor.numpy()


# class _ExpMapWrapper:
#     def __init__(self, exp_map, order):
#         self._exp_map = exp_map
#         self._order = order

#     def __call__(self, p: torch.tensor, v: torch.tensor, conn: Connection):
#         conn_coeffs = conn(p)

#         q = self._exp_map(detach_numpy(p), detach_numpy(v), detach_numpy(conn_coeffs))
#         return torch.tensor(q, dtype=p.dtype)


# class _LogMapWrapper:
#     def __init__(self, log_map):
#         self._log_map = log_map

#     def __call__(self, p: torch.tensor, q: torch.tensor, conn: Connection):
#         conn_coeffs = conn(p)

#         v = self._log_map(detach_numpy(p), detach_numpy(q), detach_numpy(conn_coeffs))
#         return torch.tensor(v, dtype=p.dtype)


class ExpMethod(Enum):
    IVP = ivp_exp_map
    APPROX_O1 = ApproxExpMapWrapper(approx_exp_map_o1)
    APPROX_O2 = ApproxExpMapWrapper(approx_exp_map_o2)
    APPROX_O3 = ApproxExpMapWrapper(approx_exp_map_o3)

    def __call__(self, p: torch.Tensor, v: torch.Tensor, conn: Connection, alpha: float = 1.0):
        return self.value(p, v, conn, alpha)


class LogMethod(Enum):
    BVP = bvp_log_map
    # SHOOTING = shooting_log_map
    APPROX_O1 = ApproxLogMapWrapper(approx_log_map_o1)
    APPROX_O2 = ApproxLogMapWrapper(approx_log_map_o2)
    APPROX_O3 = ApproxLogMapWrapper(approx_log_map_o3)

    def __call__(self, p: torch.Tensor, q: torch.Tensor, conn: Connection, alpha: float = 1.0):
        return self.value(p, q, conn, alpha)

# usable maps


# def exp_map(p, v, conn, method=ExpMethod.APPROX_O2):
#     return method(p, v, conn)
#
#
# def log_map(p, q, conn, method=LogMethod.APPROX_O2):
#     return method(p, q, conn)
#
#
# def dist_map(
#         p, q, metric: MetricField, conn: Connection = None, log_method=LogMethod.APPROX_O2
# ):
#     # if connection not defined then use the levi-civita connection from the metric
#     if conn is None:
#         conn = metric.christoffels()
#
#     # convenient method to compute this value
#     v = log_method(p, q, conn)
#     return metric(p)(v, v)

# class DistSquaredMap(Function):
#     # by defining the distance map as a torch function then we can evaluate the
#     # differential (cotangent space) of the function using the torch autograd
#     # system (we consider torch gradient to be the differential here as torch
#     # was not built for differential geometry) automatically and can therefore
#     # easily define the cost/constraint functions for an optimization problem
#     # NOTE: do not use this as part of a training pipeline

#     generate_vmap_rule = True

#     @staticmethod
#     def forward(
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

#         return dist_sqr

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         (p, q, metric, conn, log_method) = inputs

#         if conn is None:
#             conn = metric.christoffels()

#         g = metric(p)
#         v = log_method(p, q, conn)  # tangent space at p

#         dv = g.flat(v)  # cotangent space at p (differential)
#         diff_dist_sqr = -2 * dv

#         ctx.save_for_backward(diff_dist_sqr)

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

#     # @staticmethod
#     # def setup_context(ctx, inputs, output):
#     #     # all work done in the forward pass
#     #     pass

#     # @staticmethod
#     # def forward(
#     #     ctx,
#     #     p: torch.Tensor,
#     #     q: torch.Tensor,
#     #     metric: MetricField,
#     #     conn: Connection = None,
#     #     log_method=LogMethod.APPROX_SO,
#     # ):
#     #     print(f"p_requires_grad={p.requires_grad}")
#     #     print(f"q_requires_grad={q.requires_grad}")

#     #     g = metric(p)  # metric at point p
#     #     v = log_method(p, q, conn)  # tangent space at p

#     #     print(f"v: {v.requires_grad}")

#     #     dv = g.flat(v)  # cotangent space at p (differential)
#     #     diff_dist_sqr = -2 * dv

#     #     diff_dist_sqr.requires_grad_()

#     #     # diff_dist_sqr.requires_grad=True

#     #     ctx.save_for_backward(diff_dist_sqr)

#     #     dist_sqr = g(v, v) ** 2

#     #     print(f"dist_sqr: {dist_sqr.requires_grad}")

#     #     dist_sqr.requires_grad_()
#     #     print(f"dist_sqr: {dist_sqr.requires_grad}")

#     #     return dist_sqr + 0 * p.sum()

#     # @staticmethod
#     # def backward(ctx, grad_output):
#     #     # dv is already the differential of the distance
#     #     (diff_dist_sqr,) = ctx.saved_tensors
#     #     return (
#     #         grad_output * diff_dist_sqr,
#     #         # torch needs a "gradient" for each input but these are just
#     #         # various parameters so we return None (non-differentiable)
#     #         None,
#     #         None,
#     #         None,
#     #         None,
#     #         None,
#     #     )


# def dist_squared_map(
#     p: torch.Tensor,
#     q: torch.Tensor,
#     metric: MetricField,
#     conn: Connection = None,
#     log_method=LogMethod.APPROX_SO,
# ) -> torch.tensor:
#     # if a custom connection is not defined then use the levi-civita
#     # connection derived from the metric
#     if conn is None:
#         conn = metric.christoffels()
#     return DistSquaredMap.apply(p, q, metric, conn, log_method)
