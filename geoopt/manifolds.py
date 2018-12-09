import abc
import torch
from . import util

__all__ = ["Manifold", "Rn", "Stiefel"]


class Manifold(metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def check_point(self, x):
        """
        Check if point is valid to be used with the manifold

        Parameters
        ----------
        x : tensor

        Returns
        -------
        boolean indicating if tensor is valid
        """
        raise NotImplementedError

    def retr(self, x, u, t):
        """
        Perform a retraction from point with given direction and time

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
        Perform vector transport from point `x`, direction `u` and time `t` for vector `v`

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
        Inner product for tangent vectors at point x

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
        Project vector u on a tangent space

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
        Project point x on the manifold

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

    def _transp_many(self, x, u, t, *vs):
        new_vs = []
        for v in vs:
            new_vs.append(self._transp_one(x, u, t, v))
        return tuple(new_vs)

    def _retr_transp(self, x, u, t, v, *more):
        out = (self.retr(x, u, t),)
        if more:
            out = out + self._transp_many(x, u, t, v, *more)
        else:
            out = out + (self._transp_one(x, u, t, v),)
        return out

    @abc.abstractmethod
    def _retr(self, x, u, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _transp_one(self, x, u, t, v):
        raise NotImplementedError

    @abc.abstractmethod
    def _inner(self, x, u, v):
        raise NotImplementedError

    @abc.abstractmethod
    def _proju(self, x, u):
        raise NotImplementedError

    @abc.abstractmethod
    def _projx(self, x):
        raise NotImplementedError

    def __repr__(self):
        return self.name + " manifold"

    def __eq__(self, other):
        return type(self) is type(other)


class Rn(Manifold):
    """
    An unconstrained manifold
    """

    name = "Rn"
    ndim = 0
    reversible = True

    def check_point(self, x):
        return True

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
    R"""
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

    def check_point(self, x):
        return x.dim() >= 2 and x.shape[-1] <= x.shape[-2]

    def amat(self, x, u, project=True):
        if project:
            u = self.proju(x, u)
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def _proju(self, x, u):
        p = -0.5 * x @ x.transpose(-1, -2)
        p[..., range(x.shape[-2]), range(x.shape[-2])] += 1
        return p @ u

    def _projx(self, x):
        U, d, V = util.svd(x)
        return torch.einsum("...ik,...k,...jk->...ij", [U, torch.ones_like(d), V])

    def _retr(self, x, u, t):
        a = self.amat(x, u, project=False)
        rhs = x + t / 2 * a @ x
        lhs = -t / 2 * a
        lhs[..., range(a.shape[-2]), range(x.shape[-2])] += 1
        qx, _ = torch.gesv(rhs, lhs)
        return qx

    def _inner(self, x, u, v):
        return (u * v).sum([-1, -2])

    def _transp_one(self, x, u, t, v):
        a = self.amat(x, u, project=False)
        rhs = v + t / 2 * a @ v
        lhs = -t / 2 * a
        lhs[..., range(a.shape[-2]), range(x.shape[-2])] += 1
        qv, _ = torch.gesv(rhs, lhs)
        return qv
