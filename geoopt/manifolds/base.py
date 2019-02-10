from collections import defaultdict
import abc
import torch
import re

__all__ = ["Manifold"]


class ManifoldMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # get the default order for the class
        if cls._retr_transp_default_preference not in {"follow", "2y"}:
            raise RuntimeError(
                "class attribute _retr_transp_default_preference should be in {'follow', '2y'}"
            )
        default_order = cls._default_order
        # create dict access for orders for a range of methods
        retractoins = MethodDict()
        retractoins_transport = MethodDict()
        transports_follow = MethodDict()
        # loop for all class members
        for name in dir(cls):
            # retraction pattern
            if re.match(r"^_retr(\d+)?$", name):
                order = int(name[len("_retr") :] or "1")
                meth = getattr(cls, name)
                retractoins[order] = meth
            # retraction + transport pattern
            elif re.match(r"^_retr(\d+)?_transp$", name):
                meth = getattr(cls, name)
                # if some method is not implemented it should be set as `not_implemented`
                # function from exactply this module
                # to make metaclass work properly
                order = int(name[len("_retr") : -len("_transp")] or "1")
                retractoins_transport[order] = meth
            # exponential map pattern
            elif re.match(r"^_expmap$", name):
                meth = getattr(cls, name)
                retractoins[-1] = meth
            # exponential map + transport pattern
            elif re.match(r"^_expmap_transp$", name):
                meth = getattr(cls, name)
                retractoins_transport[-1] = meth
            # transport using retraction pattern
            elif re.match(r"^_transp_follow(\d+)?$", name):
                meth = getattr(cls, name)
                order = int(name[len("_transp_follow") :] or "1")
                transports_follow[order] = meth
            # transport using expmap pattern
            elif re.match(r"^_transp_follow_expmap$", name):
                meth = getattr(cls, name)
                transports_follow[-1] = meth
        # set best possible retraction to use in expmap as a fallback
        if transports_follow[-1] is not_implemented:
            best_transport_follow = max(
                (o for o, m in transports_follow.items() if m is not not_implemented),
                default=None,
            )
        else:
            best_transport_follow = -1
        if retractoins[-1] is not_implemented:
            best_retraction = max(
                (o for o, m in retractoins.items() if m is not not_implemented),
                default=None,
            )
        else:
            best_retraction = -1
        if retractoins_transport[-1] is not_implemented:
            best_retraction_transport = max(
                (
                    o
                    for o, m in retractoins_transport.items()
                    if m is not not_implemented
                ),
                default=None,
            )
        else:
            best_retraction_transport = -1
        # assign default methods
        retractoins[None] = retractoins[default_order]
        retractoins_transport[None] = retractoins_transport[default_order]
        transports_follow[None] = transports_follow[default_order]
        # assign best possible options
        retractoins[-1] = retractoins[best_retraction]
        retractoins_transport[-1] = retractoins_transport[best_retraction_transport]
        transports_follow[-1] = transports_follow[best_transport_follow]
        # set class attributes
        cls._transport_follow_funcs = transports_follow
        cls._retr_transport_funcs = retractoins_transport
        cls._retr_funcs = retractoins
        # set best options for methods in terms of approximation
        cls._best_retraction_transport = best_retraction_transport
        cls._best_retraction = best_retraction
        cls._best_transport_follow = best_transport_follow
        return cls


class MethodDict(defaultdict):
    def __missing__(self, key):
        return not_implemented


def not_implemented(*args, **kwargs):
    """
    A placeholder for not implemented methods in the Manifold
    """
    raise NotImplementedError


class Manifold(metaclass=ManifoldMeta):
    r"""
    Base class for Manifolds

    Every subclass should provide its :attr:`name`, :attr:`ndim`,
    indicate if it is :attr:`reversible`
    and implement the following:

    * ``_check_point(x, ...)`` required
        Checks point has valid dims, shapes, etc
    * ``_check_point_on_manifold(x)`` required
        Checks point lies on manifold
    * ``_check_vector_on_tangent(x, u)`` required
        Checks vector lies on tangent space to :math:`x`
    * ``_projx(x)`` required
        Projects :math:`x` on manifold
    * ``_proju(x, u)`` required
        Projects :math:`u` on tangent space at point :math:`x`, usually the same as ``_egrad2rgrad``
    * ``_inner(x, u, v)`` required
        Computes inner product :math:`\langle u, v\rangle_x`
    * ``_retr(x, u, t)`` required
        Performs retraction map for :math:`x` with direction :math:`u` and time :math:`t`
    * ``_transp_one(x, u, t, v)`` required
        Performs vector transport for :math:`v` with direction :math:`u` and time :math:`t`
    * ``_transp_many(x, u, t, *vs)`` desired
        Same as ``_transp_one(x, u, t, v)`` with multiple inputs
    * ``_retr_transp(x, u, t, *vs)`` desired
        Combines ``_transp_many(x, u, t, *vs)`` and ``_retr(x, u, t)``
    * ``__eq__(other)`` if needed
        Checks if manifolds are the same
    * ``_egrad2rgrad(u)`` if differs
        Transforms euclidean grad to Riemannian gradient.

    Notes
    -----
    Public documentation, private implementation design is used
    """
    name = None
    ndim = None
    reversible = None
    _default_order = 1

    def broadcast_scalar(self, t):
        """
        Broadcast scalar t for manifold, appending last dimensions if needed

        Parameters
        ----------
        t : scalar
            Potentially batched (individual for every point in a batch) scalar for points on the manifold.

        Returns
        -------
        scalar
            broadcasted representation for ``t``
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
            point on the manifold
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
            point on the manifold
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
            vector on the tangent plane
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
            vector on the tangent plane
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
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
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
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
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
            point on the manifold
        u : tensor
            vector on the tangent space to ``x``
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
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
            point on the manifold
        u : tensor
            vector on the tangent space to ``x``
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
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

    def retr(self, x, u, t=1.0, order=None):
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
        order : int
            order of retraction approximation, by default uses the simplest that is usually a first order approximation.
            Possible choices depend on a concrete manifold and -1 stays for exponential map

        Returns
        -------
        tensor
            transported point
        """
        t = self.broadcast_scalar(t)
        return self._retr_funcs[order](self, x, u, t)

    def expmap(self, x, u, t=1.0):
        """
        Perform an exponential map from point :math:`x` with
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
            transported point

        Notes
        -----
        By default, no error is raised if exponential map is not implemented. If so,
        the best approximation to exponential map is applied instead.
        """
        t = self.broadcast_scalar(t)
        return self._retr_funcs[-1](self, x=x, u=u, t=t)

    def expmap_transp(self, x, v, *more, u, t=1.0):
        """
        Perform an exponential map from point :math:`x` with
        given direction :math:`u` and time :math:`t`

        Parameters
        ----------
        x : tensor
            point on the manifold
        v : tensor
            tangent vector at point x to be transported
        more : tensors
            other tangent vectors at point x to be transported
        u : tensor
            tangent vector at point x
        t : scalar
            time to go with direction u

        Returns
        -------
        tensor
            transported point

        Notes
        -----
        By default, no error is raised if exponential map is not implemented. If so,
        the best approximation to exponential map is applied instead.
        """
        t = self.broadcast_scalar(t)
        return self._retr_transport_funcs[-1](self, x, v, *more, u=u, t=t)

    def transp(self, x, v, *more, u=None, t=1.0, y=None, order=None):
        """
        Perform vector transport from point :math:`x` for vector :math:`v` using one of the following:

        1. Go by direction :math:`u` and time :math:`t`
        2. Use target point :math:`y` directly

        Either :math:`y` or :math:`u` should present but not both

        Parameters
        ----------
        x : tensor
            point on the manifold
        v : tensor
            tangent vector at point x to be transported
        more : tensors
            other tangent vectors at point x to be transported
        u : tensor
            tangent vector at point x (required if :math:`y` is not provided)
        t : scalar
            time to go with direction u
        y : tensor
            the target point for vector transport  (required if :math:`u` is not provided)
        order : int
            order of retraction approximation, by default uses the simplest that is usually a first order approximation.
            Possible choices depend on a concrete manifold and -1 stays for exponential map
            This argument is used only if :math:`u` is provided

        Returns
        -------
        tensor or tuple of tensors
            transported tensor(s)
        """
        t = self.broadcast_scalar(t)
        if y is not None and u is not None:
            raise TypeError("transp() accepts either y or u only, not both")
        if y is not None:
            return self._transp2y(x, v, *more, y=y)
        elif u is not None:
            return self._transport_follow_funcs[order](self, x, v, *more, u=u, t=t)
        else:
            raise TypeError("transp() requires either y or u")

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
        scalar
            inner product (broadcasted)
        """
        if v is None and self._inner_autofill:
            v = u
        return self._inner(x, u, v)

    # dev: autofill None parameter or propagate None?
    _inner_autofill = True

    def proju(self, x, u):
        """
        Project vector :math:`u` on a tangent space for :math:`x`, usually is the same as :meth:`egrad2rgrad`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            vector to be projected

        Returns
        -------
        tensor
            projected vector
        """
        return self._proju(x, u)

    def egrad2rgrad(self, x, u):
        """
        Embed euclidean gradient into Riemannian manifold

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            gradient to be projected

        Returns
        -------
        tensor
            grad vector in the Riemainnian manifold
        """
        return self._egrad2rgrad(x, u)

    def projx(self, x):
        """
        Project point :math:`x` on the manifold

        Parameters
        ----------
        x : tensor
            point to be projected

        Returns
        -------
        tensor
            projected point
        """
        return self._projx(x)

    def retr_transp(self, x, v, *more, u, t=1.0, order=None):
        """
        Perform a retraction + vector transport at once

        Parameters
        ----------
        x : tensor
            point on the manifold
            tangent vector at point x
        t : scalar
            time to go with direction u
        v : tensor
            tangent vector at point x to be transported (required keyword only argument)
        more : tensors
            other tangent vector at point x to be transported
        u : tensor
        order : int
            order of retraction approximation, by default uses the simplest.
            Possible choices depend on a concrete manifold and -1 stays for exponential map

        Returns
        -------
        tuple of tensors
            transported point and vectors

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport
        """
        return self._retr_transport_funcs[order](self, x, v, *more, u=u, t=t)

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
            point on the manifold
        name : str
            name to be present in errors

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
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
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
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
            check result and the reason of fail if any
        """
        # return True, None
        raise NotImplementedError

    def _retr_transp(self, x, v, *more, u, t):
        """
        Developer Guide

        Naive implementation for retraction and
        transporting many vectors at once.
        """

        y = self._retr(x, u, t)
        if self._retr_transp_default_preference == "follow":
            if more:
                out = (y, ) + self._transp_follow(x, v, *more, u=u, t=t)
            else:
                out = (y, self._transp_follow(x, v, *more, u=u, t=t),)
        else:
            if more:
                out = (y, ) + self._transp2y(x, v, *more, y=y)
            else:
                out = (y, self._transp2y(x, v, *more, y=y),)
        return out

    _retr_transp_default_preference = "follow"

    @abc.abstractmethod
    def _retr(self, x, u, t):
        """
        Developer Guide

        Private implementation for retraction map. Should allow broadcasting.
        """
        raise NotImplementedError

    # def _transp_follow(self, x, v, *more, u, t):
    """
    Developer Guide

    Private implementation for vector transport using :math:`u` and :math:`t`. Should allow broadcasting.
    """
    _transp_follow = not_implemented

    # def _transp2y(self, x, v, *more, y):
    """
    Developer Guide

    Private implementation for vector transport using :math:`y`. Should allow broadcasting.
    """
    _transp2y = not_implemented

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

    def _egrad2rgrad(self, x, u):
        """
        Developer Guide

        Private implementation for gradient transformation, may do things efficiently in some cases.
        Should allow broadcasting.
        """
        return self._proju(x, u)

    def extra_repr(self):
        return ""

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + "({}) manifold".format(extra)
        else:
            return self.name + " manifold"

    def __eq__(self, other):
        return type(self) is type(other)
