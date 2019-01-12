import torch

from .base import Manifold


class Sphere(Manifold):
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
