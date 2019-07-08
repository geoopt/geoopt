import torch
from .base import Manifold
from ..utils import strip_tuple, size2shape
import geoopt


__all__ = ["Euclidean", "R"]


class R(Manifold):
    """
    Simple Euclidean manifold, every coordinate is treated as an independent element
    """

    name = "R"
    ndim = 0
    reversible = True

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        return True, None

    def retr(self, x, u):
        return x + u

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            return u.pow(2)
        else:
            return u * v

    def proju(self, x, u):
        return u

    def projx(self, x):
        return x

    def logmap(self, x, y):
        return y - x

    def dist(self, x, y, *, keepdim=False):
        return (x - y).abs()

    def egrad2rgrad(self, x, u):
        return u

    def expmap(self, x, u):
        return x + u

    def transp(self, x, y, v):
        return v

    def random_normal(self, *size, mean=0.0, std=1.0, device=None, dtype=None):
        """
        Method to create a point on the manifold, measure is induced by Normal distribution

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


class Euclidean(R):
    """
    Simple Euclidean manifold, every row is treated as an independent element
    """

    ndim = 1
    name = "Euclidean"

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def norm(self, x, u, *, keepdim=False):
        return u.norm(dim=-1)

    def dist(self, x, y, *, keepdim=False):
        return (x - y).norm(dim=-1, keepdim=keepdim)
