from .base import Manifold
from ..utils import strip_tuple


__all__ = ["Euclidean", "R"]


class R(Manifold):
    """
    Simple Euclidean manifold
    """

    name = "Euclidean"
    ndim = 0
    reversible = True

    def _check_shape(self, x, name):
        return True, None

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _retr(self, x, u):
        return x + u

    def _inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            return u.pow(2)
        else:
            return u * v

    def _proju(self, x, u):
        return u

    def _projx(self, x):
        return x

    def _transp_follow_expmap(self, x, u, v, *more):
        return strip_tuple((v, *more))

    _transp_follow_retr = _transp_follow_expmap

    def _logmap(self, x, y):
        return y - x

    def _dist(self, x, y, *, keepdim=False):
        return (x - y).abs()

    def _expmap_transp(self, x, u, v, *more):
        return (x + u, v, *more)

    _retr_transp = _expmap_transp

    def _egrad2rgrad(self, x, u):
        return u

    def _expmap(self, x, u):
        return x + u

    def _transp(self, x, y, v, *more):
        return strip_tuple((v, *more))


class Euclidean(R):
    ndim = 1

    def _check_shape(self, x, name):
        dim_is_ok = x.dim() >= 1
        if not dim_is_ok:
            return False, "Not enough dimensions for `{}`".format(name)
        return True, None

    def _inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def _dist(self, x, y, *, keepdim=False):
        return (x - y).norm(dim=-1, keepdim=keepdim)
