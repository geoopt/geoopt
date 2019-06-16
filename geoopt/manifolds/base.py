import abc
import torch.nn

__all__ = ["Manifold"]


class Manifold(torch.nn.Module):
    name = None
    ndim = None
    reversible = None

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *input):
        # this removes all warnings about implementing abstract methods
        raise TypeError("Manifold is not callable")

    def check_point(self, x, *, explain=False):
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

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
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

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(x, "x")
        if not ok:
            raise ValueError(
                "`x` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector(self, u, *, explain=False):
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

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
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

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(u, "u")
        if not ok:
            raise ValueError(
                "`u` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_point_on_manifold(self, x, *, explain=False, atol=1e-5, rtol=1e-5):
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

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x, "x")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
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

    def check_vector_on_tangent(
        self, x, u, *, ok_point=False, explain=False, atol=1e-5, rtol=1e-5
    ):
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
        ok_point: bool
            is a check for point required?

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        if not ok_point:
            ok, reason = self._check_shape(x, "x")
            if ok:
                ok, reason = self._check_shape(u, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector_on_tangent(
        self, x, u, *, ok_point=False, atol=1e-5, rtol=1e-5
    ):
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
        ok_point: bool
            is a check for point required?
        """
        if not ok_point:
            ok, reason = self._check_shape(x, "x")
            if ok:
                ok, reason = self._check_shape(u, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`u` seems to be a tensor "
                "not lying on tangent space to `x` for {} manifold.\nerror: {}".format(
                    self.name, reason
                )
            )

    def dist(self, x, y, *, keepdim=False):
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

    def retr(self, x, u):
        """
        Perform a retraction from point :math:`x` with
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
        """
        return self._retr(x, u)

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
        return self._expmap(x, u)

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

    def expmap_transp(self, x, u, v, *more):
        """
        Perform an exponential map from point :math:`x` with
        given direction :math:`u`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vectors at point :math:`x` to be transported

        Returns
        -------
        tensor
            transported point

        Notes
        -----
        By default, no error is raised if exponential map is not implemented. If so,
        the best approximation to exponential map is applied instead.
        """
        raise self._expmap_transp(x, u, v, *more)

    def transp_follow_retr(self, x, u, v, *more):
        """
        Perform vector transport from point :math:`x` for vector :math:`v` following a
        retraction map using vector :math:`u`

        Either :math:`y` or :math:`u` should present but not both

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vectors at point :math:`x` to be transported

        Returns
        -------
        tensor or tuple of tensors
            transported tensor(s)
        """
        return self._transp_follow_retr(x, u, v, *more)

    def transp_follow_expmap(self, x, u, v, *more):
        """
        Perform vector transport from point :math:`x` for vector :math:`v` following a
        and exponential (best possible retraction) map using vector :math:`u`

        Either :math:`y` or :math:`u` should present but not both

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vectors at point :math:`x` to be transported

        Returns
        -------
        tensor or tuple of tensors
            transported tensor(s)
        """
        return self._transp_follow_expmap(x, u, v, *more)

    def transp(self, x, y, v, *more):
        """
        Perform vector transport from point :math:`x` for vector :math:`v` following a
        and exponential (best possible retraction) map using vector :math:`u`

        Either :math:`y` or :math:`u` should present but not both

        Parameters
        ----------
        x : tensor
            start point on the manifold
        y : tensor
            target point on the manifold
        v : tensor
            tangent vector at point :math:`x`

        more : tensors
           other tangent vectors at point :math:`x` to be transported

        Returns
        -------
        tensor or tuple of tensors
           transported tensor(s)
        """
        return self._transp(x, y, v, *more)

    def inner(self, x, u, v=None, *, keepdim=False):
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
        return self._inner(x, u, v, keepdim=keepdim)

    def norm(self, x, u, *, keepdim=False):
        """
        Norm of a tangent vector at point :math:`x`

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        scalar
            inner product (broadcasted)
        """
        raise self._norm(x, u, keepdim=keepdim)

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

    def retr_transp(self, x, u, v, *more):
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
        return self._retr_transp(x, u, v, *more)

    # private implementation, public documentation design

    @abc.abstractmethod
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

    @abc.abstractmethod
    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
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

    @abc.abstractmethod
    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
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

    @abc.abstractmethod
    def _transp_follow_expmap(self, x, u, v, *more):
        """
        Developer Guide

        Private implementation for vector transport following an exponential map. Should allow broadcasting.
        If not existent, should fall back to the best implemented analog with retraction.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _transp_follow_retr(self, x, u, v, *more):
        """
        Developer Guide

        Private implementation for vector transport following an exponential map. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retr_transp(self, x, u, v, *more):
        """
        Developer Guide

        Private implementation for vector transport combined with retraction map. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _expmap_transp(self, x, u, v, *more):
        """
        Developer Guide

        Private implementation for vector transport combined with an exponential map. Should allow broadcasting.
        """
        raise NotImplementedError

    def _dist(self, x, y, *, keepdim=False):
        raise NotImplementedError

    @abc.abstractmethod
    def _retr(self, x, u):
        """
        Developer Guide

        Private implementation for retraction map. Should allow broadcasting.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _inner(self, x, u, v=None, *, keepdim=False):
        """
        Developer Guide

        Private implementation for inner product. Should allow broadcasting.
        """
        raise NotImplementedError

    def _norm(self, x, u, *, keepdim=False):
        """
        Developer Guide

        Private implementation for vector norm. Should allow broadcasting.
        """
        return self._inner(x, u, u, keepdim=keepdim)

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

    @abc.abstractmethod
    def _egrad2rgrad(self, x, u):
        """
        Developer Guide

        Private implementation for gradient transformation, may do things efficiently in some cases.
        Should allow broadcasting.
        """
        raise NotImplementedError

    def _logmap(self, x, y):
        """
        Developer Guide

        Private implementation for logarithmic map. May be empty
        """
        raise NotImplementedError

    def _transp(self, x, y, v, *more):
        """
        Developer Guide

        Private implementation for vector transport. May be empty
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _expmap(self, x, u):
        """
        Developer Guide

        Private implementation for exponential map. If does not exist, should fall back to best possible retraction map.
        """
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + "({}) manifold".format(extra)
        else:
            return self.name + " manifold"
