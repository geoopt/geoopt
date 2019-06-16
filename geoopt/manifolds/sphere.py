import torch

from .base import Manifold
from ..utils import strip_tuple
import geoopt.linalg.batch_linalg

__all__ = [
    "Sphere",
    "SphereSubspaceIntersection",
    "SphereSubspaceComplementIntersection",
    "SphereExact",
    "SphereSubspaceIntersectionExact",
    "SphereSubspaceComplementIntersectionExact",
]


class Sphere(Manifold):
    r"""
    Sphere manifold induced by the following constraint

    .. math::

        \|x\|=1
    """

    ndim = 1
    name = "Sphere"
    reversible = False

    def _check_shape(self, x, name):
        dim_is_ok = x.dim() >= 1
        if not dim_is_ok:
            return False, "Not enough dimensions for `{}`".format(name)
        return True, None

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        norm = x.norm(dim=-1)
        ok = torch.allclose(norm, norm.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if not ok:
            return False, "`norm(x) != 1` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        inner = self._inner(None, x, u, keepdim=True)
        ok = torch.allclose(inner, inner.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`<x, u> != 0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _inner(self, x, u, v=None, *, keepdim=False):
        return (u * v).sum(-1, keepdim=keepdim)

    def _projx(self, x):
        return x / x.norm(dim=-1, keepdim=True)

    def _proju(self, x, u):
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def _expmap(self, x, u):
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self._projx(x + u)
        cond = norm_u > 1e-3
        return torch.where(cond, exp, retr)

    def _retr(self, x, u):
        return self._projx(x + u)

    def _transp_follow_retr(self, x, u, v, *more):
        y = self._retr(x, u)
        return self._transp(x, y, v, *more)

    def _transp(self, x, y, v, *more, strip=True):
        result = tuple(self._proju(y, _v) for _v in (v,) + more)
        if strip:
            return strip_tuple(result)
        else:
            return result

    def _transp_follow_expmap(self, x, u, v, *more):
        y = self._expmap(x, u)
        return self._transp(x, y, v, *more)

    def _expmap_transp(self, x, u, v, *more):
        y = self._expmap(x, u)
        vs = self._transp(x, y, v, *more, strip=False)
        return (y,) + vs

    def _retr_transp(self, x, u, v, *more):
        y = self._retr(x, u)
        vs = self._transp(x, y, v, *more, strip=False)
        return (y,) + vs

    def _logmap(self, x, y):
        u = self._proju(x, y - x)
        dist = self._dist(x, y, keepdim=True)
        # If the two points are "far apart", correct the norm.
        cond = dist.gt(1e-6)
        return torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True), u)

    def _dist(self, x, y, *, keepdim=False):
        inner = self._inner(None, x, y, keepdim=keepdim).clamp(-1, 1)
        return torch.acos(inner)

    def _egrad2rgrad(self, x, u):
        return self._proju(x, u)


class SphereExact(Sphere):
    _retr_transp = Sphere._expmap_transp
    _transp_follow_retr = Sphere._transp_follow_expmap
    _retr = Sphere._expmap


class SphereSubspaceIntersection(Sphere):
    r"""
    Sphere manifold induced by the following constraint

    .. math::

        \|x\|=1\\
        x \in \mathbb{span}(U)

    Parameters
    ----------
    span : matrix
        the subspace to intersect with.  Shape: (..., dim, numplanes)
    """

    name = "SphereSubspace"

    def __init__(self, span):
        super().__init__()
        self._configure_manifold(span)
        if (geoopt.linalg.batch_linalg.matrix_rank(self._projector) == 1).any():
            raise ValueError(
                "Manifold only consists of isolated points when "
                "subspace is 1-dimensional."
            )

    def _check_shape(self, x, name):
        ok, reason = super()._check_shape(x, name)
        if ok:

            ok = x.shape[-1] == self._projector.shape[-2]
            if not ok:
                reason = "The leftmost shape of `span` does not match `x`: {}, {}".format(
                    x.shape[-1], self._projector.shape[-1]
                )
            elif x.dim() < (self._projector.dim() - 1):
                reason = "`x` should have at least {} dimensions but has {}".format(
                    self._projector.dim() - 1, x.dim()
                )
            else:
                reason = None
        return ok, reason

    def _configure_manifold(self, span):
        Q, _ = geoopt.linalg.batch_linalg.qr(span)
        self.register_buffer("_projector", Q @ Q.transpose(-1, -2))

    def _project_on_subspace(self, x):
        return x @ self._projector.transpose(-1, -2)

    def _proju(self, x, u):
        u = super()._proju(x, u)
        return self._project_on_subspace(u)

    def _projx(self, x):
        x = self._project_on_subspace(x)
        return super()._projx(x)


class SphereSubspaceIntersectionExact(SphereExact, SphereSubspaceIntersection):
    pass


class SphereSubspaceComplementIntersection(SphereSubspaceIntersection):
    r"""
    Sphere manifold induced by the following constraint

    .. math::

        \|x\|=1\\
        x \in \mathbb{span}(U)

    Parameters
    ----------
    span : matrix
        the subspace to compliment (being orthogonal to). Shape: (..., dim, numplanes)
    """

    def _configure_manifold(self, span):
        Q, _ = geoopt.linalg.batch_linalg.qr(span)
        P = -Q @ Q.transpose(-1, -2)
        P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1
        self.register_buffer("_projector", P)


class SphereSubspaceComplementIntersectionExact(
    SphereExact, SphereSubspaceComplementIntersection
):
    pass
