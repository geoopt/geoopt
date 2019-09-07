import torch
from .base import Manifold
from ..utils import size2shape
import geoopt


__all__ = ["Euclidean"]


class Euclidean(Manifold):
    """
    Simple Euclidean manifold, every coordinate is treated as an independent element.

    Parameters
    ----------
    ndim : int
        number of trailing dimensions treated as manifold dimensions. All the operations acting on cuch
        as inner products, etc will respect the :attr:`ndim`.
    """

    name = "Euclidean"
    ndim = 0
    reversible = True

    def __init__(self, ndim=0):
        super().__init__()
        self.ndim = ndim

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        return True, None

    def retr(self, x, u):
        return x + u

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        if self.ndim > 0:
            return inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return inner

    def norm(self, x, u, *, keepdim=False):
        if self.ndim > 0:
            return u.norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return u.abs()

    def proju(self, x, u):
        return u

    def projx(self, x):
        return x

    def logmap(self, x, y):
        return y - x

    def dist(self, x, y, *, keepdim=False):
        if self.ndim > 0:
            return (x - y).norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).abs()

    def egrad2rgrad(self, x, u):
        return u

    def expmap(self, x, u):
        return x + u

    def transp(self, x, y, v):
        return v

    def random_normal(self, *size, mean=0.0, std=1.0, device=None, dtype=None):
        """
        Create a point on the manifold, measure is induced by Normal distribution.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=device, dtype=dtype)
        tens = std.new_empty(*size).normal_() * std + mean
        return geoopt.ManifoldTensor(tens, manifold=self)

    def extra_repr(self):
        return "ndim={}".format(self.ndim)
