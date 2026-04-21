from __future__ import annotations  # remove in 3.14

import torch
import inspect

from torch.func import jacrev
from torch.autograd.functional import jacobian

from src.diff_mfld.geometry.connection import Connection


# utility function
def _coords(p):
    return [p[i] for i in range(len(p))]


class LeviCivitaConnection(Connection):
    def __init__(self, n, fn):
        super().__init__(n, n)
        self.fn = fn

    def _eval(self, p):
        return self.fn(p)


def _eval_christoffels(n, metric_fn, p: torch.Tensor) -> torch.Tensor:
    # p.requires_grad_(True)
    # print(f"p: {p}")

    g = metric_fn(p)
    g_inv = g.inverse()
    g_partials = jacrev(lambda p_jacob: metric_fn(p_jacob))(p)

    # print(f"g_partials = {g_partials[:, :, 0]}")
    # assert False

    # g_partials = jacobian(lambda p_jacob: metric_fn(p_jacob), p, create_graph=True)  # adds index at end due to partials

    # computes the connection coefficients of the Levi-Civita connection using the metric
    # print(f"g: {g}")
    # print(f"g_inv: {g_inv}")
    # print(f"g_partials: {g_partials}")
    conn_coeffs = 0.5 * (
            torch.einsum("lr,rjk->ljk", g_inv, g_partials)
            + torch.einsum("lr,rkj->ljk", g_inv, g_partials)
            - torch.einsum("lr,jkr->ljk", g_inv, g_partials)
    )
    return conn_coeffs


# technically a (0,2)-tensor but defined with additional operations to allow
# raising/lowering indices (include this as composable behavior when writing
# the computational differential geometry library later)
class MetricField:
    def __init__(self, fn, n=None):
        # allow defining a custom number of dimensions if there is a mismatch
        # between the function and the true dimension
        self.n = n if n is not None else len(inspect.getfullargspec(fn).args)

        # NOTE: compiling the equation yields a longer first call but is very
        # fast in all subsequent calls through the optimization process so this
        # is a potential strategy
        self.fn = fn

    def christoffels(self) -> Connection:
        # a function for the metric is needed here as the function is then
        # differentiated to take the jacobian when evaluating the various
        # christoffel symbols as part of the levi-civita connection
        g_mat_fn = lambda p: self.fn(*_coords(p))
        # g_inv_mat_fn = lambda p: torch.inverse(g_mat_fn(p))

        return LeviCivitaConnection(
            self.n,
            # torch.compile(
            lambda p: _eval_christoffels(self.n, g_mat_fn, p),
            # ),
        )

    @property
    def partials(self, order: int = 1) -> PartialsWrapper:
        # gets a function that generates the partials of the metric tensor as
        # a function for every point on the manifold
        return PartialsWrapper(self)

    def __call__(self, p: torch.Tensor) -> MetricView:
        metric = Metric(self, self.fn(*_coords(p)))
        return MetricView(metric, False)


# not caring here but when writing the rust library at a future point in time
# then we will want to have mutable and immutable views

# noinspection PyProtectedMember
class MetricView:
    def __init__(self, metric, inv):
        self._metric: Metric = metric
        self._inv: bool = inv

        # force generating of the inverse if an inverse view is created
        if inv:
            self._metric._create_inv()

    @property
    def field(self):
        return self._metric._field

    @property
    def inv(self):
        return MetricView(self._metric, not self._inv)

    @property
    def mat(self):
        return self._metric._matrix if not self.inv else self._metric._inv_matrix

    def __getitem__(self, slice):
        return self._metric._matrix[slice] if not self._inv else self._metric._inv_matrix[slice]

    def __call__(self, u, v):
        # either takes inner product between tangent vector or covectors
        # depending on whether this view is the metric or inverse metric
        g = self._metric._matrix if not self.inv else self._metric._inv_matrix
        return u @ g @ v

    def sharp(self, u):
        # raises cotangent vector and returns isomorphic tangent vector
        return self._metric._sharp(u)

    def flat(self, u):
        # lowers tangent vector and returns isomorphic cotangent covector
        return self._metric._flat(u)


class PartialsWrapper:
    def __init__(self, field: MetricField):
        self._field = field

    def __call__(self, p) -> torch.Tensor:
        # computes the partials of the metric tensor at point p with index of
        # the basis of differentiation specified as the last dimension
        partials = jacrev(lambda p_jac: self._field.fn(*_coords(p_jac)))(p)  # index of diff in first dim
        return torch.transpose(torch.transpose(partials, 0, 2), 0, 1)


# used internally
class Metric:
    def __init__(self, field, matrix):
        self._field = field  # the metric field itself

        self._matrix = matrix  # evaluation of field (PyTorch tensor for autograd)
        self._inv_matrix = None  # lazily compute this

    def _create_inv(self):
        if self._inv_matrix is None:
            self._inv_matrix = torch.inverse(self._matrix)

    def _sharp(self, u):
        self._create_inv()
        return self._inv_matrix @ u

    def _flat(self, u):
        return self._matrix @ u


class RnMetricField(MetricField):
    def __init__(self, n):
        super().__init__(lambda *_: torch.eye(n), n=n)
