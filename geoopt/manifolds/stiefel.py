import torch

from geoopt import linalg
from .base import Manifold


__all__ = ["Stiefel", "EuclideanStiefel", "CanonicalStiefel"]


_stiefel_doc = r"""
    Manifold induced by the following matrix constraint:

    .. math::

        X^\top X = I\\
        X \in \mathrm{R}^{n\times m}\\
        n \ge m
"""


class Stiefel(Manifold):
    __doc__ = r"""
    {}

    Parameters
    ----------
    canonical : bool
        Use canonical inner product instead of euclidean one (defaults to canonical)
    """.format(
        _stiefel_doc
    )
    ndim = 2

    def __new__(cls, canonical=True):
        if cls is Stiefel:
            if canonical:
                return super().__new__(CanonicalStiefel)
            else:
                return super().__new__(EuclideanStiefel)
        else:
            return super().__new__(cls)

    def _check_shape(self, x, name):
        dim_is_ok = x.dim() >= 2
        if not dim_is_ok:
            return False, "Not enough dimensions for `{}`".format(name)
        shape_is_ok = x.shape[-1] <= x.shape[-2]
        if not shape_is_ok:
            return (
                False,
                "`{}` should have shape[-1] <= shape[-2], got {} </= {}".format(
                    name, x.shape[-1], x.shape[-2]
                ),
            )
        return True, None

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5):
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _amat(self, x, u):
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def _projx(self, x):
        U, d, V = linalg.batch_linalg.svd(x)
        return torch.einsum("...ik,...k,...jk->...ij", U, torch.ones_like(d), V)


class CanonicalStiefel(Stiefel):
    __doc__ = r"""Stiefel Manifold with Canonical inner product

    {}
    """.format(
        _stiefel_doc
    )

    name = "Stiefel(canonical)"
    reversible = True

    def _inner(self, x, u, v):
        # <u, v>_x = tr(u^T(I-1/2xx^T)v)
        # = tr(u^T(v-1/2xx^Tv))
        # = tr(u^Tv-1/2u^Txx^Tv)
        # = tr(u^Tv-1/2u^Txx^Tv)
        # = tr(u^Tv)-1/2tr(x^Tvu^Tx)
        # = \sum_ij{(u*v}_ij}-1/2\sum_ij{(x^Tv * x^Tu)_ij}
        xtu = x.transpose(-1, -2) @ u
        if v is None:
            xtv = xtu
            v = u
        else:
            xtv = x.transpose(-1, -2) @ v
        return (u * v).sum([-1, -2]) - 0.5 * (xtv * xtu).sum([-1, -2])

    # we do faster on inner without autofill
    _inner_autofill = False

    def _transp_follow_one(self, x, v, *, u, t):
        a = self._amat(x, u)
        rhs = v + t / 2 * a @ v
        lhs = -t / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qv, _ = torch.gesv(rhs, lhs)
        return qv

    def _transp_follow_many(self, x, *vs, u, t):
        """
        An optimized transp_many for Stiefel Manifold
        """
        n = len(vs)
        vs = torch.cat(vs, -1)
        qvs = self._transp_follow_one(x, vs, u=u, t=t).view(
            x.shape[:-1] + (-1, x.shape[-1])
        )
        return tuple(qvs[..., i, :] for i in range(n))

    def _transp_follow(self, x, v, *more, u, t):
        if more:
            return self._transp_follow_many(x, v, *more, u=u, t=t)
        else:
            return self._transp_follow_one(x, v, u=u, t=t)

    def _retr_transp(self, x, v, *more, u, t):
        """
        An optimized retr_transp for Stiefel Manifold
        """
        n = 2 + len(more)
        xvs = torch.cat((x, v) + more, -1)
        qxvs = self._transp_follow_one(x, xvs, u=u, t=t).view(
            x.shape[:-1] + (-1, x.shape[-1])
        )
        return tuple(qxvs[..., i, :] for i in range(n))

    def _proju(self, x, u):
        return u - x @ u.transpose(-1, -2) @ x

    def _retr(self, x, u, t):
        return self._transp_follow_one(x, x, u=u, t=t)

    def _rand_(self, x):
        x.set_(linalg.qr(x)[0])
        return x


class EuclideanStiefel(Stiefel):
    __doc__ = r"""Stiefel Manifold with Euclidean inner product

    {}
    """.format(
        _stiefel_doc
    )

    name = "Stiefel(euclidean)"
    reversible = False

    def _proju(self, x, u):
        return u - x @ linalg.batch_linalg.sym(x.transpose(-1, -2) @ u)

    def _transp_follow(self, x, v, *more, u, t):
        y = self._retr(x, u, t)
        return self._transp2y(x, v, *more, y=y)

    def _transp2y(self, x, v, *more, y):
        if not more:
            return self._proju(y, v)
        else:
            return tuple(self._proju(y, v_) for v_ in (v,) + more)

    def _retr_transp(self, x, v, *more, u, t):
        y = self._retr(x, u, t)
        vs = self._transp2y(x, v, *more, y=y)
        if more:
            return (y,) + vs
        else:
            return y, vs

    def _inner(self, x, u, v):
        return (u * v).sum([-1, -2])

    def _retr(self, x, u, t):
        q, r = linalg.batch_linalg.qr(x + u * t)
        unflip = linalg.batch_linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q

    def _expmap(self, x, u, t):
        u = u * t
        xtu = x.transpose(-1, -2) @ u
        utu = u.transpose(-1, -2) @ u
        eye = torch.zeros_like(utu)
        eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
        logw = linalg.block_matrix([[xtu, -utu], [eye, xtu]])
        w = linalg.expm(logw)
        z = torch.cat((linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
        y = torch.cat((x, u), dim=-1) @ w @ z
        return y

    def _expmap_transp(self, x, v, *more, u, t):
        y = self._expmap(x, u, t)
        vs = self._transp2y(x, v, *more, y=y)
        if more:
            return (y,) + vs
        else:
            return y, vs

    def _transp_follow_expmap(self, x, v, *more, u, t):
        y = self._expmap(x, u, t)
        return self._transp2y(x, v, *more, y=y)

    def _rand_(self, x):
        x.set_(linalg.qr(x)[0])
        return x
