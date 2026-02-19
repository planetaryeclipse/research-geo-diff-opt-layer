import torch
from abc import abstractmethod, ABC


class Connection(ABC):
    def __init__(self, n, r):
        self._n = n  # dimension of the underlying space
        self._r = r  # dimension of the bundle vector space

    @property
    def n(self) -> int:
        return self._n

    @property
    def r(self) -> int:
        return self._r

    @abstractmethod
    def _eval(self, p):
        pass

    def __call__(self, *coords):
        # in the case of product manifolds then we merge the coordinates
        p = torch.cat(*coords, dim=0)
        coords_n = p.shape[0]
        if coords_n != self.n:
            raise ValueError(
                "Coordinates passed to connection does not match dimension of "
                f"underlying manifold: n={self.n}, coords_n={coords_n}"
            )

        return self._eval(p)
