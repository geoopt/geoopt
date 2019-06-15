import abc
import torch.nn

__all__ = ["Manifold"]


class Manifold(torch.nn.Module):
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
        Performs retraction map for :math:`x` with direction :math:`u`
    * ``_transp_follow(x, v, *more, u, t)`` required
        Performs vector transport for :math:`v` from :math:`x` with direction :math:`u`
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
            or order not in self._transport_follow_funcs
        ):
            possible_orders = (
                set(self._retr_transport_funcs)
                & set(self._retr_funcs)
                & set(self._transport_follow_funcs)
            )
            raise ValueError(
                "new default order should be one of {}".format(possible_orders)
            )
        self._retr_transport_funcs = self._retr_transport_funcs.copy()
        self._retr_transport_funcs[None] = self._retr_transport_funcs[order]
        self._retr_funcs = self._retr_funcs.copy()
        self._retr_funcs[None] = self._retr_funcs[order]
        self._transport_follow_funcs = self._transport_follow_funcs.copy()
        self._transport_follow_funcs[None] = self._transport_follow_funcs[order]
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

    def retr(self, x, u, *, order=None):
        """
        Perform a retraction from point :math:`x` with
        given direction :math:`u`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        order : int
            order of retraction approximation, by default uses the simplest that is usually a first order approximation.
            Possible choices depend on a concrete manifold and -1 stays for exponential map

        Returns
        -------
        tensor
            transported point
        """
        return self._retr_funcs[order](self, x, u)

    def expmap(self, x, u):
        """
        Perform an exponential map from point :math:`x` with
        given direction :math:`u`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`

        Returns
        -------
        tensor
            transported point

        Notes
        -----
        By default, no error is raised if exponential map is not implemented. If so,
        the best approximation to exponential map is applied instead.
        """
        return self._retr_funcs[-1](self, x=x, u=u)

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

    def expmap_transp(self, x, v, *more, u):
        """
        Perform an exponential map from point :math:`x` with
        given direction :math:`u`

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

        Returns
        -------
        tensor
            transported point

        Notes
        -----
        By default, no error is raised if exponential map is not implemented. If so,
        the best approximation to exponential map is applied instead.
        """
        return self._retr_transport_funcs[-1](self, x, v, *more, u=u)

    def transp(self, x, v, *more, u=None, y=None, order=None):
        """
        Perform vector transport from point :math:`x` for vector :math:`v` using one of the following:

        1. Go by direction :math:`u`
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
        if y is not None and u is not None:
            raise TypeError("transp() accepts either y or u only, not both")
        if y is not None:
            return self._transp2y(x, v, *more, y=y)
        elif u is not None:
            return self._transport_follow_funcs[order](self, x, v, *more, u=u)
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

    def retr_transp(self, x, v, *more, u, order=None):
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
        return self._retr_transport_funcs[order](self, x, v, *more, u=u)

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
        a given point lies in the tangent space at x
        of the manifold. It should return a boolean
        indicating whether the test was passed
        and a reason of failure if check is not passed. 
        You can assume assert_check_point is already
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

    def _retr_transp(self, x, v, *more, u):
        """
        Developer Guide

        Naive implementation for retraction and
        transporting many vectors at once.
        """

        y = self._retr(x, u)
        if self._retr_transp_default_preference == "follow":
            if more:
                out = (y,) + self._transp_follow(x, v, *more, u=u)
            else:
                out = (y, self._transp_follow(x, v, *more, u=u))
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
    def _retr(self, x, u):
        """
        Developer Guide

        Private implementation for retraction map. Should allow broadcasting.
        """
        raise NotImplementedError

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
