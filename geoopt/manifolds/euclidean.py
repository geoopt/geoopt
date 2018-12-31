from .base import Manifold

__all__ = ["Euclidean"]


class Euclidean(Manifold):
    """
    Simple Euclidean manifold
    """

    name = "Euclidean"
    ndim = 0
    reversible = True

    def _check_shape(self, x, name):
        return True, None

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5):
        return True, None

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
