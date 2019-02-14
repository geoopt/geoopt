import torch

from .base import Manifold
import geoopt.linalg.batch_linalg

__all__ = [
    "Sphere",
    "SphereSubspaceIntersection",
    "SphereSubspaceComplementIntersection",
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
    _retr_transp_default_preference = "2y"

    def _check_shape(self, x, name):
        dim_is_ok = x.dim() >= 1
        if not dim_is_ok:
            return False, "Not enough dimensions for `{}`".format(name)
        return True, None

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        norm = x.norm(dim=-1)
        ok = torch.allclose(norm, norm.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if not ok:
            return False, "`norm(x) != 1` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5):
        inner = self._inner(None, x, u)
        ok = torch.allclose(inner, inner.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`<x, u> != 0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _inner(self, x, u, v):
        return (u * v).sum(-1)

    def _projx(self, x):
        return x / x.norm(dim=-1, keepdim=True)

    def _proju(self, x, u):
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    def _expmap(self, x, u, t):
        ut = u * t
        norm_ut = ut.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_ut) + ut * torch.sin(norm_ut) / norm_ut
        retr = self._projx(x + ut)
        cond = norm_ut > 1e-3
        return torch.where(cond, exp, retr)

    def _retr(self, x, u, t):
        return self._projx(x + u * t)

    def _transp_follow(self, x, v, *more, u, t):
        y = self._retr(x, u, t)
        return self._transp2y(x, v, *more, y=y)

    def _transp2y(self, x, v, *more, y):
        if more:
            return tuple(self._proju(y, _v) for _v in (v,) + more)
        else:
            return self._proju(y, v)

    def _transp_follow_expmap(self, x, v, *more, u, t):
        y = self._expmap(x, u, t)
        return self._transp2y(x, v, *more, y=y)

    def _expmap_transp(self, x, v, *more, u, t):
        y = self._expmap(x, u, t)
        vs = self._transp2y(x, v, *more, y=y)
        if more:
            return (y,) + vs
        else:
            return y, vs

    def _logmap(self, x, y):
        u = self._proju(x, y - x)
        dist = self._dist(x, y).unsqueeze(-1)
        # If the two points are "far apart", correct the norm.
        cond = dist.gt(1e-6)
        return torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True), u)

    def _dist(self, x, y):
        inner = self._inner(None, x, y).clamp(-1, 1)
        return torch.acos(inner)

    def _rand_(self, x):
        x.normal_()
        x.set_(self._projx(x))
        return x


class SphereSubspaceIntersection(Sphere):
    r"""
    Sphere manifold induced by the following constraint

    .. math::

        \|x\|=1\\
        x \in \mathbb{span}(U)

    Parameters
    ----------
    span : matrix
        the subspace to intersect with
    """

    name = "SphereSubspace"

    def __init__(self, span):
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
        self._projector = Q @ Q.transpose(-1, -2)

    def _project_on_subspace(self, x):
        return x @ self._projector.transpose(-1, -2)

    def _proju(self, x, u):
        u = super()._proju(x, u)
        return self._project_on_subspace(u)

    def _projx(self, x):
        x = self._project_on_subspace(x)
        return super()._projx(x)


class SphereSubspaceComplementIntersection(SphereSubspaceIntersection):
    r"""
    Sphere manifold induced by the following constraint

    .. math::

        \|x\|=1\\
        x \in \mathbb{span}(U)

    Parameters
    ----------
    span : matrix
        the subspace to compliment (being orthogonal to)
    """

    def _configure_manifold(self, span):
        Q, _ = geoopt.linalg.batch_linalg.qr(span)
        P = -Q @ Q.transpose(-1, -2)
        P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1
        self._projector = P
