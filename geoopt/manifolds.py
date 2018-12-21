import abc
import torch
from . import util

__all__ = ["Manifold", "Euclidean", "Stiefel"]


class Manifold(metaclass=abc.ABCMeta):
    R"""
    Base class for Manifolds

    Every subclass should provide its `name`, `ndim`,
    indicate if it is `reversible`
    and implement the following:

    * :meth:`_check_point(x)` if needed
        Checks point has valid dims, shapes, etc
    * :meth:`_check_point_on_manifold(x)` if needed
        Checks point lies on manifold
    * :meth:`_projv(x)` required
        Projects :math:`x` on manifold
    * :meth:`_proju(x, u)` required
        Projects :math:`u` on tangent space at point :math:`x`
    * :meth:`_inner(x, u, v)` required
        Computes inner product :math:`\langle u, v\rangle_x`
    * :meth:`_retr(x, u, t)` required
        Performs retraction map for :math:`x` with direction :math:`u` and time :math:`t`
    * :meth:`_transp_one(x, u, t, v)` required
        Performs vector transport for :math:`v` with direction :math:`u` and time :math:`t`
    * :meth:`_transp_many(x, u, t, *vs)` desired
        Same as :meth:`_transp_one(x, u, t, v)` with multiple inputs
    * :meth:`_retr_transp(x, u, t, *vs)` desired
        Combines :meth:`_transp_many(x, u, t, *vs)` and :meth:`_retr(x, u, t)`
    * :meth:`__eq__(other)` if needed

    Notes
    -----
    Public documentation, private implementation design is used
    """
    name = ""
    ndim = 0
    reversible = False

    def broadcast_scalar(self, t):
        """
        Broadcast scalar t for manifold, appending last dimensions if needed

        Parameters
        ----------
        t : scalar

        Returns
        -------
        scalar

        Notes
        -----
        scalar can be batch sized
        """
        if isinstance(t, torch.Tensor):
            extra = (1,) * self.ndim
            t = t.view(t.shape + extra)
        return t

    def check_point(self, x, explain=False):
        """
        Check if point is valid to be used with the manifold

        Parameters
        ----------
        x : tensor
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        ok, reason = self._check_point(x)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point(self, x):
        """
        Check if point is valid to be used with the manifold and
        raise an error with informative message on failure

        Parameters
        ----------
        x : tensor
        """

        ok, reason = self._check_point(x)
        if not ok:
            raise ValueError(
                '`x` seems to be not valid '
                'tensor for {} manifold.\nerror: {}'.format(self.name, reason)
            )

    def check_point_on_manifold(self, x, explain=False, atol=1e-5, rtol=1e-5):
        """
        Check if point :math:`x` is lying on the the manifold

        Parameters
        ----------
        x : tensor
        atol: float
        rtol: float
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        ok, reason = self._check_point(x)
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        """
        Check if point is lying on the the manifold and
        raise an error with informative message on failure

        Parameters
        ----------
        x : tensor
        atol: float
        rtol: float
        """
        self.assert_check_point(x)
        ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                '`x` seems to be a tensor '
                'not lying on {} manifold.\nerror: {}'.format(self.name, reason)
            )

    def retr(self, x, u, t):
        """
        Perform a retraction from point :math:`x` with
        given direction :math:`u` and time :math:`t`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point x
        t : scalar
            time to go with direction u

        Returns
        -------
        tensor
            new_x
        """
        t = self.broadcast_scalar(t)
        return self._retr(x, u, t)

    def transp(self, x, u, t, v, *more):
        """
        Perform vector transport from point :math:`x`,
        direction :math:`xu` and time :math:`t` for vector :math:`v`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point x
        t : scalar
            time to go with direction u
        v : tensor
            tangent vector at point x to be transported
        more : tensor
            other tangent vector at point x to be transported

        Returns
        -------
        transported tensors
        """
        t = self.broadcast_scalar(t)
        if more:
            return self._transp_many(x, u, t, v, *more)
        else:
            return self._transp_one(x, u, t, v)

    def inner(self, x, u, v=None):
        """
        Inner product for tangent vectors at point :math:`x`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point x
        v : tensor (optional)
            tangent vector at point x

        Returns
        -------
        inner product (broadcasted)

        """
        if v is None:
            v = u
        return self._inner(x, u, v)

    def proju(self, x, u):
        """
        Project vector :math:`u` on a tangent space for :math:`x`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            vector to be projected

        Returns
        -------
        projected vector
        """
        return self._proju(x, u)

    def projx(self, x):
        """
        Project point :math:`x` on the manifold

        Parameters
        ----------
        x : tensor
            point to be projected

        Returns
        -------
        projected point
        """
        return self._projx(x)

    def retr_transp(self, x, u, t, v, *more):
        """
        Perform a retraction + vector transport at once

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point x
        t : scalar
            time to go with direction u
        v : tensor
            tangent vector at point x to be transported
        more : tensor
            other tangent vector at point x to be transported

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport

        Returns
        -------
        tuple of tensors
            (new_x, *new_vs)
        """
        return self._retr_transp(x, u, t, v, *more)

    # private implementation, public documentation design

    def _check_point(self, x):
        """
        Developer Guide

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It should return boolean and
        a reason of failure if check is not passed

        Parameters
        ----------
        x : tensor

        Returns
        -------
        bool, str
        """
        return True, None

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        """
        Developer Guide

        Exhaustive implementation for checking if
        a given point lies on the manifold. It
        should return boolean and a reason of
        failure if check is not passed. You can
        assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x : tensor
        atol : float
            absolute tolerance
        rtol :
            relative tolerance
        Returns
        -------
        bool, str or None
        """
        return True, None

    def _transp_many(self, x, u, t, *vs):
        """
        Developer Guide

        Naive implementation for transporting many vectors at once.
        """
        new_vs = []
        for v in vs:
            new_vs.append(self._transp_one(x, u, t, v))
        return tuple(new_vs)

    def _retr_transp(self, x, u, t, v, *more):
        """
        Developer Guide

        Naive implementation for retraction and
        transporting many vectors at once.
        """

        out = (self.retr(x, u, t),)
        if more:
            out = out + self._transp_many(x, u, t, v, *more)
        else:
            out = out + (self._transp_one(x, u, t, v),)
        return out

    @abc.abstractmethod
    def _retr(self, x, u, t):
        """
        Developer Guide

        Private implementation for retraction map. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _transp_one(self, x, u, t, v):
        """
        Developer Guide

        Private implementation for vector transport. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _inner(self, x, u, v):
        """
        Developer Guide

        Private implementation for inner product. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _proju(self, x, u):
        """
        Developer Guide

        Private implementation for vector projection. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _projx(self, x):
        """
        Developer Guide

        Private implementation for point projection. Should allow broadcasting.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.name + " manifold"

    def __eq__(self, other):
        return type(self) is type(other)

class Euclidean(Manifold):
    """
    Euclidean manifold

    An unconstrained manifold
    """

    name = "Euclidean"
    ndim = 0
    reversible = True

    def _retr(self, x, u, t):
        return x + t * u

    def _inner(self, x, u, v):
        return u * v

    def _proju(self, x, u):
        return u

    def _projx(self, x):
        return x

    def _transp_one(self, x, u, t, v):
        return v


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

    def _check_point(self, x):
        dim_is_ok = x.dim() >= 2
        if not dim_is_ok:
            return False, 'Not enough dimensions'
        shape_is_ok = x.shape[-1] <= x.shape[-2]
        if not shape_is_ok:
            return False, ('Should be shape[-1] <= shape[-2], got {} </= {}'
                           .format(x.shape[-1], x.shape[-2]))
        return True, None

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _amat(self, x, u, project=True):
        if project:
            u = self.proju(x, u)
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def _proju(self, x, u):
        p = -0.5 * x @ x.transpose(-1, -2)
        p[..., torch.arange(x.shape[-2]), torch.arange(x.shape[-2])] += 1
        return p @ u

    def _projx(self, x):
        U, d, V = util.svd(x)
        return torch.einsum("...ik,...k,...jk->...ij", [U, torch.ones_like(d), V])

    def _retr(self, x, u, t):
        a = self._amat(x, u, project=False)
        rhs = x + t / 2 * a @ x
        lhs = -t / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qx, _ = torch.gesv(rhs, lhs)
        return qx

    def _inner(self, x, u, v):
        return (u * v).sum([-1, -2])

    def _transp_one(self, x, u, t, v):
        a = self._amat(x, u, project=False)
        rhs = v + t / 2 * a @ v
        lhs = -t / 2 * a
        lhs[..., torch.arange(a.shape[-2]), torch.arange(x.shape[-2])] += 1
        qv, _ = torch.gesv(rhs, lhs)
        return qv

    def _transp_many(self, x, u, t, *vs):
        """
        An optimized transp_many for Stiefel Manifold
        """
        n = len(vs)
        vs = torch.cat(vs, -1)
        qv = self._transp_one(x, u, t, vs).view(*x.shape[:-1], -1, x.shape[-1])
        return tuple(qv[..., i, :] for i in range(n))

    def _retr_transp(self, x, u, t, v, *more):
        """
        An optimized retr_transp for Stiefel Manifold
        """
        n = 2 + len(more)
        xvs = torch.cat((x, v) + more, -1)
        qxvs = self._transp_one(x, u, t, xvs).view(*x.shape[:-1], -1, x.shape[-1])
        return tuple(qxvs[..., i, :] for i in range(n))
