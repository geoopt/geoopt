import abc
import torch

__all__ = ["Manifold"]


class Manifold(metaclass=abc.ABCMeta):
    r"""
    Base class for Manifolds

    Every subclass should provide its `name`, `ndim`,
    indicate if it is `reversible`
    and implement the following:

    * :meth:`_check_point(x, ...)` required
        Checks point has valid dims, shapes, etc
    * :meth:`_check_point_on_manifold(x)` required
        Checks point lies on manifold
    * :meth:`_check_vector_on_tangent(x, u)` required
        Checks vector lies on tangent space to :math:`x`
    * :meth:`_projx(x)` required
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
    name = None
    ndim = None
    reversible = None

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
        ok, reason = self._check_shape(x, "x")
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

        ok, reason = self._check_shape(x, "x")
        if not ok:
            raise ValueError(
                "`x` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector(self, u, explain=False):
        """
        Check if point is valid to be used with the manifold

        Parameters
        ----------
        u : tensor
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        ok, reason = self._check_shape(u, "u")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector(self, u):
        """
        Check if point is valid to be used with the manifold and
        raise an error with informative message on failure

        Parameters
        ----------
        u : tensor
        """

        ok, reason = self._check_shape(u, "u")
        if not ok:
            raise ValueError(
                "`u` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_point_on_manifold(self, x, explain=False, atol=1e-5, rtol=1e-5):
        """
        Check if point :math:`x` is lying on the manifold

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
        ok, reason = self._check_shape(x, "x")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        """
        Check if point is lying on the manifold and
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
                "`x` seems to be a tensor "
                "not lying on {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector_on_tangent(self, x, u, explain=False, atol=1e-5, rtol=1e-5):
        """
        Check if u :math:`u` is lying on the tangent space to x

        Parameters
        ----------
        x : tensor
        u : tensor
        atol: float
        rtol: float
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        ok, reason = self._check_shape(x, "x")
        if ok:
            ok, reason = self._check_shape(u, "u")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5):
        """
        Check if u :math:`u` is lying on the tangent space to x and raise an error on fail

        Parameters
        ----------
        x : tensor
        u : tensor
        atol: float
        rtol: float
        """
        ok, reason = self._check_shape(x, "x")
        if ok:
            ok, reason = self._check_shape(u, "u")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`u` seems to be a tensor "
                "not lying on tangent space to `x` for {} manifold.\nerror: {}".format(
                    self.name, reason
                )
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
        if v is None and self._inner_autofill:
            v = u
        return self._inner(x, u, v)

    # dev: autofill None parameter or propagate None?
    _inner_autofill = True

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

    def _check_shape(self, x, name):
        """
        Developer Guide

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It should return boolean and
        a reason of failure if check is not passed

        Parameters
        ----------
        x : tensor
        name : str
            name to be present in errors

        Returns
        -------
        bool, str
        """
        # return True, None
        raise NotImplementedError

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
        # return True, None
        raise NotImplementedError

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5):
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
        u : tensor
        atol : float
            absolute tolerance
        rtol :
            relative tolerance
        Returns
        -------
        bool, str or None
        """
        # return True, None
        raise NotImplementedError

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
