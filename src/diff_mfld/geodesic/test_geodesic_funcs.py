import torch

import pytest
from pytest import approx

import itertools

from src.diff_mfld.geodesic.geodesic_funcs import (
    ExpMethod, LogMethod
)
from src.diff_mfld.geometry.metric import RnMetricField, MetricField


@pytest.mark.parametrize("methods", [
    (ExpMethod.IVP, LogMethod.BVP),
    (ExpMethod.APPROX_O1, LogMethod.APPROX_O1),
    (ExpMethod.APPROX_O2, LogMethod.APPROX_O2),
    (ExpMethod.APPROX_O3, LogMethod.APPROX_O3),
])
def test_euclidean_exp_log(methods):
    exp_method, log_method = methods

    metric_field = RnMetricField(3)
    conn = metric_field.christoffels()

    p = torch.tensor([1., 2., 3.])
    q = torch.tensor([4., -3., -1.])
    v = q - p  # expected in Euclidean space

    v_log = log_method(p, q, conn)
    assert v == approx(v_log)

    q_exp = exp_method(p, v, conn)
    assert q == approx(q_exp)


# tests noneuclidean metric spaces
# NOTE: if the metrics have a high norm then the high partials will cause instability and failure when evaluating log

def noneuclid_noncoupled_metric(x1, x2, x3):
    # elements are assigned to this metric to preserve gradient history
    metric = torch.zeros((3, 3))
    metric[0, 0] = x1 ** 2 + 1.  # constant prevents degeneracy at origin
    metric[1, 1] = x2 ** 2 + 1.
    metric[2, 2] = x3 ** 2 + 1.
    return metric


def noneuclid_coupled_metric(x1, x2, x3):
    metric = torch.zeros((3, 3))
    metric[0, 0] = 0.5 * x1 ** 2 + 1.
    metric[1, 1] = 0.25 * x1 ** 2 * x2 ** 2 + 1.
    metric[2, 2] = 0.128 * x1 ** 2 * x2 ** 2 * x3 ** 2 + 1.
    return metric


metric_funcs = [
    # NOTE: if the metrics have a high norm then the high partials will cause problems with trying to converge
    noneuclid_noncoupled_metric,
    noneuclid_coupled_metric,
]


@pytest.mark.parametrize("metric", metric_funcs)
def test_noneuclidean_exp_log(metric):
    exp_method, log_method = ExpMethod.IVP, LogMethod.BVP

    metric_field = MetricField(metric)
    conn = metric_field.christoffels()

    p = torch.tensor([1., 2., 3.])
    q = torch.tensor([4., -3., -1.])
    v_euclid = q - p  # result should not be this

    v_log = log_method(p, q, conn)
    q_exp = exp_method(p, v_log, conn)

    print(f"v_log = {v_log}")
    print(f"q_exp = {q_exp}")

    assert v_euclid != approx(v_log, rel=1e-3)
    assert q == approx(q_exp, rel=1e-3)


# noinspection DuplicatedCode
@pytest.mark.parametrize("metric", metric_funcs)
def test_noneuclidean_compare_with_o1(metric):
    exact_exp_method, exact_log_method = ExpMethod.IVP, LogMethod.BVP
    approx_exp_method, approx_log_method = ExpMethod.APPROX_O1, LogMethod.APPROX_O1  # euclidean

    metric_field = MetricField(metric)
    conn = metric_field.christoffels()

    p = torch.tensor([1., 2., 3.]) / 5.
    q = torch.tensor([4., -3., -1.]) / 5.

    exact_v = exact_log_method(p, q, conn, 0.25)
    approx_v = approx_log_method(p, q, conn, 0.25)

    print(f"exact_v = {exact_v}, approx_v = {approx_v}, diff: {exact_v - approx_v}")

    exact_q = exact_exp_method(p, exact_v, conn, 0.25)
    approx_q = approx_exp_method(p, exact_v, conn, 0.25)

    print(f"exact_q = {exact_q}, approx_q = {approx_q}, diff: {exact_q - approx_q}")


# noinspection DuplicatedCode
@pytest.mark.parametrize("metric", metric_funcs)
def test_noneuclidean_compare_with_o2(metric):
    exact_exp_method, exact_log_method = ExpMethod.IVP, LogMethod.BVP
    approx_exp_method, approx_log_method = ExpMethod.APPROX_O2, LogMethod.APPROX_O2  # euclidean

    metric_field = MetricField(metric)
    conn = metric_field.christoffels()

    p = torch.tensor([1., 2., 3.]) / 5.
    q = torch.tensor([4., -3., -1.]) / 5.

    exact_v = exact_log_method(p, q, conn, 0.25)
    approx_v = approx_log_method(p, q, conn, 0.25)

    print(f"exact_v = {exact_v}, approx_v = {approx_v}, diff: {exact_v - approx_v}")

    exact_q = exact_exp_method(p, exact_v, conn, 0.25)
    approx_q = approx_exp_method(p, exact_v, conn, 0.25)

    print(f"exact_q = {exact_q}, approx_q = {approx_q}, diff: {exact_q - approx_q}")


# noinspection DuplicatedCode
@pytest.mark.parametrize("metric", metric_funcs)
def test_noneuclidean_compare_with_o3(metric):
    exact_exp_method, exact_log_method = ExpMethod.IVP, LogMethod.BVP
    approx_exp_method, approx_log_method = ExpMethod.APPROX_O3, LogMethod.APPROX_O3  # euclidean

    metric_field = MetricField(metric)
    conn = metric_field.christoffels()

    p = torch.tensor([1., 2., 3.]) / 5.
    q = torch.tensor([4., -3., -1.]) / 5.

    exact_v = exact_log_method(p, q, conn, 0.25)
    approx_v = approx_log_method(p, q, conn, 0.25)

    print(f"exact_v = {exact_v}, approx_v = {approx_v}, diff: {exact_v - approx_v}")

    exact_q = exact_exp_method(p, exact_v, conn, 0.25)
    approx_q = approx_exp_method(p, exact_v, conn, 0.25)

    print(f"exact_q = {exact_q}, approx_q = {approx_q}, diff: {exact_q - approx_q}")
