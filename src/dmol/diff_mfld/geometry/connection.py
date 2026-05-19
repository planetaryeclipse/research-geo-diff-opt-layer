import torch

from torch.func import jacrev

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
    def _eval(self, p) -> torch.Tensor:
        pass

    def __call__(self, *coords_sets) -> torch.Tensor:
        # in the case of product manifolds then we merge the coordinates
        p = torch.cat(tuple(coords for coords in coords_sets), dim=0)
        coords_n = p.shape[0]
        if coords_n != self.n:
            raise ValueError(
                "Coordinates passed to connection does not match dimension of "
                f"underlying manifold: n={self.n}, coords_n={coords_n}"
            )

        # if multiple positions have been provided to this call (in the respective columns) then
        # they will be segmented and calls made to the internal _eval individually
        if len(p.shape) == 1:
            return self._eval(p)
        elif len(p.shape) == 2:
            num_samples = p.shape[1]
            batched_conn_coeffs = torch.zeros((self.r, self.r, self.n, num_samples))
            for idx in range(num_samples):
                p_sample = p[:, idx].squeeze()
                conn_coeffs = self._eval(p_sample)
                batched_conn_coeffs[:, :, :, idx] = conn_coeffs
            return batched_conn_coeffs
        else:
            raise ValueError(f"incorrect shape for p {p.shape}")

    def partials(self, p: torch.Tensor, order: int = 1) -> torch.Tensor:
        if order < 1:
            raise ValueError(
                "derivative order of connection coefficient partials must be greater than 0"
            )

        # print(f"order: {order}")

        fn = self._eval
        for _ in range(order):
            fn = jacrev(fn)

        partials_val = fn(p)

        # print(f"partials_val: {partials_val.shape}")

        # print(f"partials: order = {order}, value = {partials_val}")

        return partials_val
