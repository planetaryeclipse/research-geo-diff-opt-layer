import torch

from torch.func import jacrev

from typing import Callable

from diff_mfld_optim.geometry.connection import Connection


class StatespaceConnection(Connection):
    # computes the induced connection from considering the motion of the
    # statespace system to be the corresponding parallel curve resulting from
    # the integral curve of the vector field of the statespace equation

    def __init__(self, n, u, f_stspace: Callable[[torch.Tensor], torch.Tensor]):
        # NOTE: the following doesn't actually specify an ordering of the
        # product manifold corresponding to the local trivialization of the
        # IST (M x U or U x M) but for convention we will only pass in values
        # of f_stspace with in put M x U
        # TODO: when writing the rust library in future figure out some way
        # to formalize these so we can either ensure compatibility or remap
        # components if necessary

        self._ist_dim = n + u  # locally IST is states + inputs
        self._r_vs_dim = n  # vector space with dimension of state space

        super().__init__(
            self._ist_dim, self._r_vs_dim  # larger base space  # smaller v.s.
        )

        # f_stspace serves as our vector field on the n-dimensional vector
        # space R on top of the local trivialization of the IST
        self.f_stspace = f_stspace

    def _eval(self, p):
        # note that p is on the local trivialization (product manifold) of IST
        # and in the following we're using the n and r defined for the vector
        # bundle and not n as defined as the dimension of state manifold m

        conn_coeffs = torch.zeros((self.r, self.r, self.n))
        d = self.f_stspace(p).unsqueeze(1)  # as column vector

        # for efficiency
        dd_pinv = torch.pinverse(d @ d.T)

        for k in range(self.r):
            # gets the gradient for the kth component of f
            bk = torch.unsqueeze(jacrev(lambda p: self.f_stspace(p)[0])(p), 1)
            ck = -dd_pinv @ d @ bk.T

            conn_coeffs[k, :, :] = ck
        return conn_coeffs
