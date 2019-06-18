import torch

from .base import Manifold
from ..utils import strip_tuple
import geoopt.linalg.batch_linalg

__all__ = ["Sphere", "SphereExact"]

EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


class Sphere(Manifold):
    r"""
    Sphere manifold induced by the following constraint

    .. math::

        \|x\|=1
        x \in \mathbb{span}(U)

    where :math:`U` can be parametrized with compliment space or intersection.

    Parameters
    ----------
    intersection : tensor
        shape ``(..., dim, K)``, subspace to intersect with
    complement : tensor
        shape ``(..., dim, K)``, subspace to compliment
    """
    ndim = 1
    name = "Sphere"
    reversible = False

    def __init__(self, intersection=None, complement=None):
        super().__init__()
        if intersection is not None and complement is not None:
            raise TypeError(
                "Can't initialize with both intersection and compliment arguments, please specify only one"
            )
        elif intersection is not None:
            self._configure_manifold_intersection(intersection)
        elif complement is not None:
            self._configure_manifold_complement(complement)
        else:
            self._configure_manifold_no_constraints()
        if (
            self.projector is not None
            and (geoopt.linalg.batch_linalg.matrix_rank(self.projector) == 1).any()
        ):
            raise ValueError(
                "Manifold only consists of isolated points when "
                "subspace is 1-dimensional."
            )

    def _check_shape(self, x, name):
        ok = x.dim() >= 1
        if not ok:
            ok, reason = False, "Not enough dimensions for `{}`".format(name)
        else:
            ok, reason = True, None
        if ok and self.projector is not None:
            ok = x.shape[-1] == self.projector.shape[-2]
            if not ok:
                reason = "The leftmost shape of `span` does not match `x`: {}, {}".format(
                    x.shape[-1], self.projector.shape[-1]
                )
            elif x.dim() < (self.projector.dim() - 1):
                reason = "`x` should have at least {} dimensions but has {}".format(
                    self.projector.dim() - 1, x.dim()
                )
            else:
                reason = None
        return ok, reason

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        norm = x.norm(dim=-1)
        ok = torch.allclose(norm, norm.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if not ok:
            return False, "`norm(x) != 1` with atol={}, rtol={}".format(atol, rtol)
        ok = torch.allclose(self._project_on_subspace(x), x, atol=atol, rtol=rtol)
        if not ok:
            return (
                False,
                "`x` is not in the subspace of the manifold with atol={}, rtol={}".format(
                    atol, rtol
                ),
            )
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        inner = self._inner(None, x, u, keepdim=True)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False, "`<x, u> != 0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _inner(self, x, u, v=None, *, keepdim=False):
        return (u * v).sum(-1, keepdim=keepdim)

    def _projx(self, x):
        x = self._project_on_subspace(x)
        return x / x.norm(dim=-1, keepdim=True)

    def _proju(self, x, u):
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def _expmap(self, x, u):
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self._projx(x + u)
        cond = norm_u > EPS[norm_u.dtype]
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
        cond = dist.gt(EPS[dist.dtype])
        return torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True), u)

    def _dist(self, x, y, *, keepdim=False):
        inner = self._inner(None, x, y, keepdim=keepdim).clamp(-1, 1)
        return torch.acos(inner)

    def _egrad2rgrad(self, x, u):
        return self._proju(x, u)

    def _configure_manifold_complement(self, complement):
        Q, _ = geoopt.linalg.batch_linalg.qr(complement)
        P = -Q @ Q.transpose(-1, -2)
        P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1
        self.register_buffer("projector", P)

    def _configure_manifold_intersection(self, intersection):
        Q, _ = geoopt.linalg.batch_linalg.qr(intersection)
        self.register_buffer("projector", Q @ Q.transpose(-1, -2))

    def _configure_manifold_no_constraints(self):
        self.register_buffer("projector", None)

    def _project_on_subspace(self, x):
        if self.projector is not None:
            return x @ self.projector.transpose(-1, -2)
        else:
            return x


class SphereExact(Sphere):
    _retr_transp = Sphere._expmap_transp
    _transp_follow_retr = Sphere._transp_follow_expmap
    _retr = Sphere._expmap

    def extra_repr(self):
        return "exact"
