import torch

from .. import linalg
from ..utils import strip_tuple
from .base import Manifold


__all__ = ["Stiefel", "EuclideanStiefel", "CanonicalStiefel", "EuclideanStiefelExact"]


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

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _projx(self, x):
        U, _, V = linalg.batch_linalg.svd(x)
        return torch.einsum("...ik,...jk->...ij", U, V)


class CanonicalStiefel(Stiefel):
    __doc__ = r"""Stiefel Manifold with Canonical inner product

    {}
    """.format(
        _stiefel_doc
    )

    name = "Stiefel(canonical)"
    reversible = True

    @staticmethod
    def _amat(x, u):
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def _inner(self, x, u, v=None, *, keepdim=False):
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
        return (u * v).sum([-1, -2], keepdim=keepdim) - 0.5 * (xtv * xtu).sum(
            [-1, -2], keepdim=keepdim
        )

    def _transp_follow_one(self, x, v, *, u):
        a = self._amat(x, u)
        rhs = v + 1 / 2 * a @ v
        lhs = -1 / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qv, _ = torch.solve(rhs, lhs)
        return qv

    def _transp_follow_many(self, x, *vs, u):
        """
        An optimized transp_many for Stiefel Manifold
        """
        vs = torch.cat(vs, -1)
        qvs = self._transp_follow_one(x, vs, u=u).view(x.shape[:-1] + (-1, x.shape[-1]))
        return qvs.unbind(-2)

    def _transp_follow_retr(self, x, u, v, *more):
        return strip_tuple(self._transp_follow_many(x, v, *more, u=u))

    _transp_follow_expmap = _transp_follow_retr

    def _retr_transp(self, x, u, v, *more):
        """
        An optimized retr_transp for Stiefel Manifold
        """
        xvs = torch.cat((x, v) + more, -1)
        qxvs = self._transp_follow_one(x, xvs, u=u).view(
            x.shape[:-1] + (-1, x.shape[-1])
        )
        return qxvs.unbind(-2)

    _expmap_transp = _retr_transp

    def _proju(self, x, u):
        return u - x @ u.transpose(-1, -2) @ x

    _egrad2rgrad = _proju

    def _retr(self, x, u):
        return self._transp_follow_one(x, x, u=u)

    _expmap = _retr


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

    _egrad2rgrad = _proju

    def _transp(self, x, y, v, *more, strip=True):
        result = tuple(self._proju(y, _v) for _v in (v,) + more)
        if strip:
            return strip_tuple(result)
        else:
            return result

    def _retr_transp(self, x, u, v, *more):
        y = self._retr(x, u)
        vs = self._transp2y(x, v, *more, y=y)
        if more:
            return (y,) + vs
        else:
            return y, vs

    def _inner(self, x, u, v=None, *, keepdim=False):
        return (u * v).sum([-1, -2], keepdim=keepdim)

    def _retr(self, x, u):
        q, r = linalg.batch_linalg.qr(x + u)
        unflip = linalg.batch_linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q

    def _expmap(self, x, u):
        xtu = x.transpose(-1, -2) @ u
        utu = u.transpose(-1, -2) @ u
        eye = torch.zeros_like(utu)
        eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
        logw = linalg.block_matrix([[xtu, -utu], [eye, xtu]])
        w = linalg.expm(logw)
        z = torch.cat((linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
        y = torch.cat((x, u), dim=-1) @ w @ z
        return y

    def _expmap_transp(self, x, u, v, *more):
        y = self._expmap(x, u)
        vs = self._transp(x, y, v, *more, strip=False)
        return (y,) + vs

    def _transp_follow_expmap(self, x, u, v, *more):
        y = self._expmap(x, u)
        return self._transp(x, y, v, *more)

    def _transp_follow_retr(self, x, u, v, *more):
        y = self._retr(x, u)
        return self._transp(x, y, v, *more)


class EuclideanStiefelExact(EuclideanStiefel):
    _retr_transp = EuclideanStiefel._expmap_transp
    _transp_follow_retr = EuclideanStiefel._transp_follow_expmap
    _retr = EuclideanStiefel._expmap

    def extra_repr(self):
        return "exact"
