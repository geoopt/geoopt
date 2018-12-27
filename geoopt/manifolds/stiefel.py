import torch

from geoopt import util
from .base import Manifold


__all__ = ["Stiefel"]


class Stiefel(Manifold):
    r"""
    Manifold induced by the following matrix constraint:

    .. math::

        X^\top X = I
        X \in \mathrm{R}^{n\times m}
        n \ge m

    Notes
    -----
    works with batch sized tensors
    """

    name = "Stiefel"
    ndim = 2
    reversible = True

    def __init__(self, canonical=True):
        self.canonical = canonical

    def _check_shape(self, x, name):
        dim_is_ok = x.dim() >= 2
        if not dim_is_ok:
            return False, "Not enough dimensions"
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

    def _proju_canonical(self, x, u):
        return u - x @ u.transpose(-1, -2) @ x

    def _proju_euclidean(self, x, u):
        return u - x @ util.linalg.sym(x.transpose(-1, -2) @ u)

    def _proju(self, x, u):
        if self.canonical:
            return self._proju_canonical(x, u)
        else:
            return self._proju_euclidean(x, u)

    def _projx(self, x):
        U, d, V = util.linalg.svd(x)
        return torch.einsum("...ik,...k,...jk->...ij", [U, torch.ones_like(d), V])

    @staticmethod
    def _inner_euclidean(u, v):
        if v is None:
            v = u
        return (u * v).sum([-1, -2])

    @staticmethod
    def _inner_canonical(x, u, v):
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

    def _inner(self, x, u, v):
        # override the public function that
        # does some magic with setting v=u in some cases
        if self.canonical:
            # we can speed up computation
            # removing one matmul for inner product
            return self._inner_canonical(x, u, v)
        else:
            return self._inner_euclidean(u, v)

    # do not autofill, we do it by hand to speed up computations
    _inner_autofill = False

    def _transp_one_canonical(self, x, u, t, v):
        a = self._amat(x, u)
        rhs = v + t / 2 * a @ v
        lhs = -t / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qv, _ = torch.gesv(rhs, lhs)
        return qv

    def _transp_many_canonical(self, x, u, t, *vs):
        """
        An optimized transp_many for Stiefel Manifold
        """
        n = len(vs)
        vs = torch.cat(vs, -1)
        qvs = self._transp_one_canonical(x, u, t, vs).view(
            *x.shape[:-1], -1, x.shape[-1]
        )
        return tuple(qvs[..., i, :] for i in range(n))

    def _retr_transp_canonical(self, x, u, t, v, *more):
        """
        An optimized retr_transp for Stiefel Manifold
        """
        n = 2 + len(more)
        xvs = torch.cat((x, v) + more, -1)
        qxvs = self._transp_one_canonical(x, u, t, xvs).view(
            *x.shape[:-1], -1, x.shape[-1]
        )
        return tuple(qxvs[..., i, :] for i in range(n))

    def _transp_one_euclidean(self, x, u, t, v, y=None):
        if y is None:
            y = self._retr_euclidean(x, u, t)
        return self._proju_euclidean(y, v)

    def _retr_canonical(self, x, u, t):
        return self._transp_one(x, u, t, x)

    @staticmethod
    def _retr_euclidean(x, u, t):
        q, r = util.linalg.qr(x + u * t)
        unflip = torch.sign(torch.sign(util.linalg.extract_diag(r)) + 0.5)
        q *= unflip
        return q

    def _retr(self, x, u, t):
        if self.canonical:
            return self._retr_canonical(x, u, t)
        else:
            return self._retr_euclidean(x, u, t)

    def _transp_many_euclidean(self, x, u, t, *vs, y=None):
        if y is None:
            y = self._retr_euclidean(x, u, t)
        return tuple(self._proju_euclidean(y, v) for v in vs)

    def _retr_transp_euclidean(self, x, u, t, v, *more):
        y = self._retr_euclidean(x, u, t)
        vs = self._transp_many_euclidean(x, u, t, v, *more, y=y)
        return (y,) + vs

    def _retr_transp(self, x, u, t, v, *more):
        if self.canonical:
            return self._retr_transp_canonical(x, u, t, v, *more)
        else:
            return self._retr_transp_euclidean(x, u, t, v, *more)

    def _transp_one(self, x, u, t, v):
        if self.canonical:
            return self._transp_one_canonical(x, u, t, v)
        else:
            return self._transp_one_euclidean(x, u, t, v)

    def _transp_many(self, x, u, t, *vs):
        if self.canonical:
            return self._transp_many_canonical(x, u, t, *vs)
        else:
            return self._transp_many_euclidean(x, u, t, *vs)

    def __eq__(self, other):
        return super().__eq__(other) and self.canonical == other.canonical
