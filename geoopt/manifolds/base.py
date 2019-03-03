from collections import defaultdict
from bisect import bisect_right
import abc
import torch.nn

__all__ = ["Manifold"]


class ApproxMethodDecorator:
    def __new__(cls, *methods, order=1):
        self = super().__new__(cls)
        if len(methods) == 0:
            return self
        self.__init__(order=order)
        return self(*methods)

    def __init__(self, order=1):
        self.order = order

    def __call__(self, meth):
        setattr(meth, '_methodset', self.methodset)
        setattr(meth, '_methodorder', self.order)
        return meth

    @staticmethod
    def is_approx_method(method):
        try:
            ApproxMethodDecorator.get_methodset(method)
            ApproxMethodDecorator.get_order(method)
            return True
        except AttributeError:
            return False

    @staticmethod
    def get_methodset(method):
        return getattr(method, '_methodset')

    @staticmethod
    def get_order(method):
        return getattr(method, '_methodorder')



class Retraction(ApproxMethodDecorator):
    """
    Methods decorated with``@Retraction(order)``
    implement given-order approximations of exponential map.
    Exact exponential map corresponds to ``order=-1``.

    Example
    -------

    .. code-block:: python
        @Retraction
        def _retr(self, point, direction, time):
            return point + direction * time
    """
    methodset = 'retr'


class RetractAndTransport(ApproxMethodDecorator):
    """
    Methods decorated with``@RetractAndTransport(order)``
    implement ``retr_transp``.
    with given retraction ``order``.


    Example
    -------

    .. code-block:: python
        @RetractAndTransport
        def _retr_transp(
            point,
            tangent,
            *more_tangents,
            transport_direction,
            transport_time):
            pass
    """
    methodset = 'retr_transport'


class Transport(ApproxMethodDecorator):
    """
    Methods decorated with ``@Transport``
    implement ``transp()``
    """
    methodset = 'transport'


class TransportAlong(ApproxMethodDecorator):
    """
    Methods decorated with ``@Transport``
    implement ``transp_along()``,
    i.e. transport along a direction
    for a given time
    """
    methodset = 'transport_along'


class TransportAlongAndExpmap(ApproxMethodDecorator):
    """
    It's really unclear for me what these methods do,
    but I wouldn't just remove them
    """
    methodset = 'transp_along_expmap'


class ManifoldMeta(abc.ABCMeta):
    r"""
    We use a metaclass that tracks and registers implementations
    of approximate methods, e.g. approximations of exponential map (retractions),
    approximations of parallel transport, et cetera..

    During construction of a Manifold class,
    ``ManifoldMeta`` iterates over ``dir(cls)``
    and looks for methods marked by any of ``ApproxMethodDecorator``\s.
    Such methods are registered in corresponding ``MethodDict``\s.
    Best (``order=-1``) and default (``order=None``)
    approximations are provided.

    Approximations missing implementation should be registered as
    ``geoopt.base.not_implemented``, which is a placeholder function
    that raises a "not implemented" error.

    Special private functions contain the following:

    * marked by ``@Retraction``
    * marked by ``@TransportAlong``: vector transport that uses direction + retraction rather that the final point
    * marked by ``@Transport``: parallel transport
    * (TODO) marked by ``@Expmap``: same as ``@Retraction(order=-1)``
    * (TODO) marked by ``@ExpmapAndTransport``: for exponential map + vector transport (retraction with order ``-1``)
    * (TODO) marked by ``@TransportAlongAndExpmap``: vector transport that uses direction + exponential map rather that the final point (retraction+transport with order `-1`)

    .. code-block:: python

        retractions = MethodDict()
        retractions_transport = MethodDict()
        transports_follow = MethodDict()

    With this dict it comes possible to define generic dispatch methods for different orders of approximations like this:

    .. code-block:: python

        def retr(self, x, u, t=1.0, order=None):
            t = self.broadcast_scalar(t)
            return self._retr_funcs[order](self, x, u, t)

    As you see, we avoid weird code that makes use of ``if`` or ``getattr`` with handling exceptions.

    Exponential map is dispatched in the same way

    .. code-block:: python

        def expmap(self, x, u, t=1.0):
            t = self.broadcast_scalar(t)
            return self._retr_funcs[-1](self, x, u, t)

    """

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        # get the default order for the class
        if cls._retr_transp_default_preference not in {"follow", "2y"}:
            raise RuntimeError(
                "class attribute _retr_transp_default_preference should be in {'follow', '2y'}"
            )
        default_order = cls._default_order
        # create dict access for orders for a range of methods
        methodsets = defaultdict(MethodDict)
        # loop for all class members
        for name in dir(cls):
            meth = getattr(cls, name)
            if not ApproxMethodDecorator.is_approx_method(meth):
                continue
            ms = ApproxMethodDecorator.get_methodset(meth)
            order = ApproxMethodDecorator.get_order(meth)
            methodsets[ms][order] = meth
        for ms, methods in methodsets.items():
            order_cmp_key = lambda o: float('inf') if o in [-1, None] else o
            implemented_orders = sorted(methods.keys(), key=order_cmp_key)
            # set best possible retraction to use in expmap as a fallback
            if -1 not in methods:
                methods[-1] = methods[implemented_orders[-1]]
            # assign default methods
            if None not in methods:
                # TODO: check if default_order is implemented
                methods[None] = methods[default_order]
        # set class attributes
        for ms, methods in methodsets.items():
            attr_name = f'_{ms}_funcs'
            setattr(cls, attr_name, methods)
        return cls


class MethodDict(defaultdict):
    def __missing__(self, key):
        return not_implemented


def not_implemented(*args, **kwargs):
    """
    A placeholder for not implemented methods in the Manifold
    """
    raise NotImplementedError


class Manifold(torch.nn.Module, metaclass=ManifoldMeta):
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
    * ``_egrad2rgrad(u)`` if differs from ``_proju(x, u)``
        Transforms euclidean grad to Riemannian gradient.
    * ``_inner(x, u, v)`` required
        Computes inner product :math:`\langle u, v\rangle_x`
    * ``_retr(x, u, t)`` required
        Performs retraction map for :math:`x` with direction :math:`u` and time :math:`t`
    * ``_transp_follow(x, v, *more, u, t)`` required
        Performs vector transport for :math:`v` from :math:`x` with direction :math:`u` and time :math:`t`
    * ``_transp2y(x, v, *more, u, t)`` desired
        Performs vector transport for :math:`v` with from :math:`x` to :math:`y`
    * ``_retr_transp(x, v, *more, u, t)`` desired
        Combines retraction and vector transport
    * ``__eq__(other)`` if needed
        Checks if manifolds are the same

    Notes
    -----
    Public documentation, private implementation design is used.
    Some more about design info is in :class:`geoopt.manifolds.base.ManifoldMeta`.
    """
    name = None
    ndim = None
    reversible = None
    _default_order = 1

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *input):
        # this removes all warnings about implementing abstract methods
        raise TypeError("Manifold is not callable")

    # noinspection PyAttributeOutsideInit
    def set_default_order(self, order):
        """
        Set the default order of approximation. This might be useful to specify retraction being used in optimizers

        Parameters
        ----------
        order : int|None
            default order of retraction approximation (None stays for Manifold default value)

        Returns
        -------
        Manifold
            returns same instance
        """
        if order is None:
            order = type(self)._default_order
        if (
            order not in self._retr_transport_funcs
            or order not in self._retr_funcs
            or order not in self._transport_along_funcs
        ):
            possible_orders = (
                set(self._retr_transport_funcs)
                & set(self._retr_funcs)
                & set(self._transport_along_funcs)
            )
            raise ValueError(
                "new default order should be one of {}".format(possible_orders)
            )
        self._retr_transport_funcs = self._retr_transport_funcs.copy()
        self._retr_transport_funcs[None] = self._retr_transport_funcs[order]
        self._retr_funcs = self._retr_funcs.copy()
        self._retr_funcs[None] = self._retr_funcs[order]
        self._transport_along_funcs = self._transport_along_funcs.copy()
        self._transport_along_funcs[None] = self._transport_along_funcs[order]
        self._retr_funcs[None] = self._retr_funcs[order]
        self._default_order = order
        return self

    def reset_default_order(self):
        """
        Reset the default order of approximation. The new order will
        be the initial default approximation order for the manifold.

        Returns
        -------
        Manifold
            returns same instance
        """
        return self.set_default_order(None)

    @property
    def default_order(self):
        return self._default_order

    @default_order.setter
    def default_order(self, order):
        self.set_default_order(order)

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
            vector on the tangent space to :math:`x`
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
            vector on the tangent space to :math:`x`
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

    def dist(self, x, y, keepdim=False):
        """
        Compute distance between 2 points on the manifold that is the shortest path along geodesics

        Parameters
        ----------
        x : tensor
            point on the manifold
        y : tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        scalar
            distance between two points
        """
        return self._dist(x, y, keepdim=keepdim)

    def retr(self, x, u, t=1.0, order=None):
        """
        Perform a retraction from point :math:`x` with
        given direction :math:`u` and time :math:`t`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        t : scalar
            time to go with direction :math:`u`
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
            tangent vector at point :math:`x`
        t : scalar
            time to go with direction :math:`u`

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

    def logmap(self, x, y):
        r"""
        Perform an logarithmic map for a pair of points :math:`x` and :math:`y`.
        The result lies in :math:`u \in T_x\mathcal{M}` is such that:

        .. math::

            y = \operatorname{Exp}_x(\operatorname{Log}_{x}(y))

        Parameters
        ----------
        x : tensor
            point on the manifold
        y : tensor
            point on the manifold

        Returns
        -------
        tensor
            tangent vector
        """
        return self._logmap(x, y)

    def expmap_transp(self, x, v, *more, u, t=1.0):
        """
        Perform an exponential map from point :math:`x` with
        given direction :math:`u` and time :math:`t`

        Parameters
        ----------
        x : tensor
            point on the manifold
        v : tensor
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vectors at point :math:`x` to be transported
        u : tensor
            tangent vector at point :math:`x`
        t : scalar
            time to go with direction :math:`u`

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
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vectors at point :math:`x` to be transported
        u : tensor
            tangent vector at point :math:`x` (required if :math:`y` is not provided)
        t : scalar
            time to go with direction :math:`u`
        y : tensor
            the target point for vector transport  (required if :math:`u` is not provided)
        order : int
            order of retraction approximation, by default uses the simplest that is usually a first order approximation.
            Possible choices depend on a concrete manifold and -1 stays for exponential map.
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
            return self._transport_along_funcs[order](self, x, v, *more, u=u, t=t)
        else:
            raise TypeError("transp() requires either y or u")

    def inner(self, x, u, v=None, keepdim=False):
        """
        Inner product for tangent vectors at point :math:`x`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor (optional)
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        scalar
            inner product (broadcasted)
        """
        if v is None and self._inner_autofill:
            v = u
        return self._inner(x, u, v, keepdim=keepdim)

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
        v : tensor
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vector at point :math:`x` to be transported
        u : tensor
            tangent vector at point :math:`x` (required keyword only argument)
        t : scalar
            time to go with direction :math:`u`
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
                out = (y,) + self._transp_follow(x, v, *more, u=u, t=t)
            else:
                out = (y, self._transp_follow(x, v, *more, u=u, t=t))
        else:
            if more:
                out = (y,) + self._transp2y(x, v, *more, y=y)
            else:
                out = (y, self._transp2y(x, v, *more, y=y))
        return out

    """
    To make ``retr_transp`` work in case of ``_transp2y`` is much more efficient than 
    ``_transp_follow`` there is a class attribute ``_retr_transp_default_preference`` to indicate this. 
    The attribute should be present in the class definition if differs from default provided in `Manifold`.
    Its values should be in {'follow', '2y'}, default is 'follow'
    """
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

    # def _logmap(self, x, y):
    """
    Developer Guide

    Private implementation for logarithmic map for :math:`x` and :math:`y`. Should allow broadcasting.
    """
    _logmap = not_implemented

    # def _expmap(self, x, y):
    """
    Developer Guide

    Private implementation for exponential map for :math:`x` and :math:`y`. Should allow broadcasting.
    """
    _expmap = not_implemented

    # def _dist(self, x, y):
    """
    Developer Guide

    Private implementation for computing distance between :math:`x` and :math:`y`. Should allow broadcasting.
    """
    _dist = not_implemented

    @abc.abstractmethod
    def _inner(self, x, u, v, keepdim):
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
