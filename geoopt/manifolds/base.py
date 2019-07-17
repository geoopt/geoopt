import abc
import torch.nn

__all__ = ["Manifold"]


class Manifold(torch.nn.Module, metaclass=abc.ABCMeta):
    name = None
    ndim = None
    reversible = None

    forward = NotImplemented

    def __init__(self, **kwargs):
        super().__init__()

    def check_point(self, x, *, explain=False):
        """
        Check if point is valid to be used with the manifold.

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
        ok, reason = self._check_shape(x.shape, "x")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point(self, x):
        """
        Check if point is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : tensor
            point on the manifold

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(x.shape, "x")
        if not ok:
            raise ValueError(
                "`x` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector(self, u, *, explain=False):
        """
        Check if vector is valid to be used with the manifold.

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
        ok, reason = self._check_shape(u.shape, "u")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector(self, u):
        """
        Check if vector is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        u : tensor
            vector on the tangent plane

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(u.shape, "u")
        if not ok:
            raise ValueError(
                "`u` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_point_on_manifold(self, x, *, explain=False, atol=1e-5, rtol=1e-5):
        """
        Check if point :math:`x` is lying on the manifold.

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
        ok, reason = self._check_shape(x.shape, "x")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        """
        Check if point :math`x` is lying on the manifold and raise an error with informative message on failure.

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
        Check if :math:`u` is lying on the tangent space to x.

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
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
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
        Check if u :math:`u` is lying on the tangent space to x and raise an error on fail.

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
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
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
        Compute distance between 2 points on the manifold that is the shortest path along geodesics.

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
        raise NotImplementedError

    @abc.abstractmethod
    def retr(self, x, u):
        """
        Perform a retraction from point :math:`x` with given direction :math:`u`.

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
        raise NotImplementedError

    @abc.abstractmethod
    def expmap(self, x, u):
        r"""
        Perform an exponential map :math:`\operatorname{Exp}_x(u)`.

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
        raise NotImplementedError

    def logmap(self, x, y):
        r"""
        Perform an logarithmic map :math:`\operatorname{Log}_{x}(y)`.

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
        raise NotImplementedError

    @abc.abstractmethod
    def expmap_transp(self, x, u, v, *more):
        """
        Perform an exponential map and vector transport from point :math:`x` with given direction :math:`u`.

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
        raise NotImplementedError

    @abc.abstractmethod
    def transp_follow_retr(self, x, u, v, *more):
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrac{T}_{x\to\operatorname{retr}(x, u)}(v)`.

        This operation is sometimes is much more simpler and can be optimized.

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
        raise NotImplementedError

    @abc.abstractmethod
    def transp_follow_expmap(self, x, u, v, *more):
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrac{T}_{x\to\operatorname{Exp}(x, u)}(v)`.

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
        raise NotImplementedError

    def transp(self, x, y, v, *more):
        r"""
        Perform vector transport :math:`\mathfrac{T}_{x\to y}(v)`.

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
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, x, u, v=None, *, keepdim=False):
        """
        Inner product for tangent vectors at point :math:`x`.

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
        raise NotImplementedError

    def norm(self, x, u, *, keepdim=False):
        """
        Norm of a tangent vector at point :math:`x`.

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
        raise self.inner(x, u, keepdim=keepdim) ** 0.5

    @abc.abstractmethod
    def proju(self, x, u):
        """
        Project vector :math:`u` on a tangent space for :math:`x`, usually is the same as :meth:`egrad2rgrad`.

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
        raise NotImplementedError

    @abc.abstractmethod
    def egrad2rgrad(self, x, u):
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            gradient to be projected

        Returns
        -------
        tensor
            grad vector in the Riemannian manifold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x):
        """
        Project point :math:`x` on the manifold.

        Parameters
        ----------
        x : tensor
            point to be projected

        Returns
        -------
        tensor
            projected point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retr_transp(self, x, u, v, *more):
        """
        Perform an retraction and vector transport from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor
            tangent vector at point :math:`x` to be transported
        more : tensors
            other tangent vector at point :math:`x` to be transported

        Returns
        -------
        tuple of tensors
            transported point and vectors

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport
        """
        raise NotImplementedError

    def _check_shape(self, shape, name):
        """
        Util to check shape.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It should return boolean and
        a reason of failure if check is not passed

        Parameters
        ----------
        shape : tuple
            shape of point on the manifold
        name : str
            name to be present in errors

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        ok = len(shape) >= self.ndim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.ndim
            )
        else:
            reason = None
        return ok, reason

    def _assert_check_shape(self, shape, name):
        """
        Util to check shape and raise an error if needed.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It will raise a ValueError if check is not passed

        Parameters
        ----------
        shape : tuple
            shape of point on the manifold
        name : str
            name to be present in errors

        Raises
        ------
        ValueError
        """
        ok, reason = self._check_shape(shape, name)
        if not ok:
            raise ValueError(reason)

    @abc.abstractmethod
    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        """
        Util to check point lies on the manifold.

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
        Util to check a vector belongs to the tangent space of a point.

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

    def extra_repr(self):
        return ""

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + "({}) manifold".format(extra)
        else:
            return self.name + " manifold"
