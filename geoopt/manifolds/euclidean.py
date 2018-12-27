from .base import Manifold

__all__ = ["Euclidean"]


class Euclidean(Manifold):
    """
    Euclidean manifold

    An unconstrained manifold
    """

    name = "Euclidean"
    ndim = 0
    reversible = True

    def _retr(self, x, u, t):
        return x + t * u

    def _inner(self, x, u, v):
        return u * v

    def _proju(self, x, u):
        return u

    def _projx(self, x):
        return x

    def _transp_one(self, x, u, t, v):
        return v
