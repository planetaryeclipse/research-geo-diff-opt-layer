import torch
import numpy as np

import pytest
from pytest import approx

from optim.subsolvers.rtr import _project_onto_ellipsoid, solve_tr_subproblem_m


# def _project_onto_ellipsoid(x: np.ndarray, p_mat: np.nparray, b: float) -> np.ndarray:
#     # this is not the orthogonal projection, only scaling with respect to origin
#     if x @ p_mat @ x <= b:
#         return x
#     else:
#         p_sqrt = np.sqrt(p_mat)
#
#         x_transf = p_sqrt @ x
#         x_transf /= np.linalg.norm(x_transf)
#         x_transf *= np.sqrt(b)
#
#         x_proj = np.linalg.inv(p_sqrt) @ x_transf
#
#         return x_proj

def test_project_onto_ellipsoid():
    p_mat = np.array(np.diag([1.0, 1.0]))
    b = 0.9**2

    print(f"b: {b}")

    x = np.array([-0.65132156, -0.65132156])
    print(f"x: {x}, norm: {np.linalg.norm(x)}")
    proj_x = _project_onto_ellipsoid(x, p_mat, b)
    norm_sqr = proj_x @ p_mat @ proj_x

    print(f"x: {x}, proj_x: {proj_x}, norm_sqr: {norm_sqr}")


    pass

def test_solve_tr_subproblem_m():
    p = torch.tensor([2., 3.])
    grad_f_k = torch.tensor([1.0, 1.0])
    h_k = torch.eye(2)
    g_k = torch.diag(torch.tensor([1.0, 1.0]))

    radius_k = 0.5
    max_iters_k = 100
    rel_acc_k = 1e-4
    damp_k = 0.1

    eta = solve_tr_subproblem_m(p, grad_f_k, h_k, g_k, radius_k, max_iters_k, rel_acc_k, damp_k)
    print(f"eta: {eta}")


