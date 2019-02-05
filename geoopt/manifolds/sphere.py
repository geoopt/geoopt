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

    def _retr(self, x, u, t):
        ut = u * t
        norm_ut = ut.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_ut) + ut * torch.sin(norm_ut) / norm_ut
        retr = self._projx(x + ut)
        cond = norm_ut < 1e-3
        return torch.where(cond, exp, retr)

    def _transp_one(self, x, u, t, v, y=None):
        if y is None:
            y = self._retr(x, u, t)
        return self._proju(y, v)

    def _transp_many(self, x, u, t, *vs, y=None):
        if y is None:
            y = self._retr(x, u, t)
        return tuple(self._proju(y, v) for v in vs)

    def _retr_transp(self, x, u, t, v, *more):
        y = self._retr(x, u, t)
        vs = self._transp_many(x, u, t, v, *more, y=y)
        return (y,) + vs


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
