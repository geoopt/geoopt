from .base import Manifold
from .base import Retraction
from .base import RetractAndTransport
from .base import Transport
from .base import TransportAlong

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

    @Retraction
    def _retr(self, x, u, t):
        return x + t * u

    def _inner(self, x, u, v, keepdim):
        return u * v

    def _proju(self, x, u):
        return u

    def _projx(self, x):
        return x

    @TransportAlong
    def _transp_follow(self, x, v, *more, u, t):
        if not more:
            return v
        else:
            return (v,) + more

    _retr_transp_default_preference = "2y"

    @Transport
    def _transp2y(self, x, v, *more, y):
        if not more:
            return v
        else:
            return (v,) + more

    def _logmap(self, x, y):
        return y - x

    def _dist(self, x, y, keepdim):
        return (x - y).abs()
