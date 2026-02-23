import torch
import inspect
import itertools

from torch.func import jacrev
from torch.autograd.functional import jacobian

from diff_mfld_optim.geometry.connection import Connection


def _coords(p):
    return [p[i] for i in range(len(p))]


class LeviCivitaConnection(Connection):
    def __init__(self, n, fn):
        super().__init__(n, n)
        self.fn = fn

    def _eval(self, p):
        return self.fn(p)


def _eval_christoffels(n, g_fn, g_inv_fn, p) -> torch.tensor:
    # evaluate all the partials of the metric elements
    conn_coeffs = torch.zeros((n, n, n))
    g_inv = g_inv_fn(p)

    metric_partials = jacrev(g_fn)(p)
    for k, i, j in itertools.product(range(n), range(n), range(n)):
        coeff = 0.0
        for m in range(n):
            coeff += (
                0.5
                * g_inv[k, m]
                * (
                    metric_partials[m, i, j]
                    + metric_partials[m, j, i]
                    - metric_partials[i, j, m]
                )
            )
        conn_coeffs[k, i, j] = coeff
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
        g_inv_mat_fn = lambda p: torch.inverse(g_mat_fn(p))

        return LeviCivitaConnection(
            self.n,
            # torch.compile(
            lambda p: _eval_christoffels(self.n, g_mat_fn, g_inv_mat_fn, p),
            # ),
        )

    def __call__(self, p):
        metric = Metric(self, self.fn(*_coords(p)))
        return MetricView(metric, False)


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


# not caring here but when writing the rust library at a future point in time
# then we will want to have mutable and immutable views
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
        (
            self._metric._matrix[slice]
            if not self._inv
            else self._metric._inv_matrix[slice]
        )

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


class RnMetricField(MetricField):
    def __init__(self, n):
        super().__init__(lambda *_: torch.eye(n), n=n)
