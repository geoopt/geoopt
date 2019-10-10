import abc
import torch.nn
import inspect

__all__ = ["Manifold", "ScalingInfo"]


class ScalingInfo(object):
    """
    Scaling info for each argument that requires rescaling.

    .. code:: python

        scaled_value = value * scaling ** power if power != 0 else value

    For results it is not always required to set powers of scaling, then it is no-op
    """

    __slots__ = ["kwargs", "results"]

    def __init__(self, *results: float, **kwargs: float):
        self.results = results
        self.kwargs = kwargs


class ScalingStorage(dict):
    """
    Helper class to make implementation transparent.
    """

    def __call__(self, scaling_info: ScalingInfo):
        def register(fn):
            self[fn.__name__] = scaling_info
            return fn

        return register


class Manifold(torch.nn.Module, metaclass=abc.ABCMeta):
    __scaling__ = ScalingStorage()  # will be filled along with implementation below
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

    @__scaling__(ScalingInfo(1))
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

    @__scaling__(ScalingInfo(2))
    def dist2(self, x, y, *, keepdim=False):
        """
        Compute squared distance between 2 points on the manifold that is the shortest path along geodesics.

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
            squared distance between two points
        """
        raise self.dist(x, y, keepdim=keepdim) ** 2

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
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
    @__scaling__(ScalingInfo(u=-1))
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
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(1))
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

    @__scaling__(ScalingInfo(u=-1))
    def expmap_transp(self, x, u, v):
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

        Returns
        -------
        tensor
            transported point
        """
        y = self.expmap(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        """
        Perform a retraction + vector transport at once.

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        tuple of tensors
            transported point and vectors

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport
        """
        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_retr(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
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

        Returns
        -------
        tensor
            transported tensor
        """
        y = self.retr(x, u)
        return self.transp(x, y, v)

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrac{T}_{x\to\operatorname{Exp}(x, u)}(v)`.

        Here, :math:`\operatorname{Exp}` is the best possible approximation of the true exponential map.
        There are cases when the exact variant is hard or impossible implement, therefore a
        fallback, non-exact, implementation is used.

        Parameters
        ----------
        x : tensor
            point on the manifold
        u : tensor
            tangent vector at point :math:`x`
        v : tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        tensor
            transported tensor
        """
        y = self.expmap(x, u)
        return self.transp(x, y, v)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor):
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

        Returns
        -------
        tensor or tuple of tensors
           transported tensor(s)
        """
        raise NotImplementedError

    @abc.abstractmethod
    @__scaling__(ScalingInfo(2))
    def inner(self, x: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False):
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

    @__scaling__(ScalingInfo(1))
    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False):
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
    def proju(self, x: torch.Tensor, u: torch.Tensor):
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
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor):
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
    def projx(self, x: torch.Tensor):
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
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
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
    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
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

    def unpack_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Construct a point on the manifold.

        This method should help to work with product and compound manifolds.
        Internally all points on the manifold are stored in an intuitive format.
        However, there might be cases, when this representation is simpler or more efficient to store in
        a different way that is hard to use in practice.

        Returns
        -------
        torch.Tensor
        """
        return tensor

    def pack_point(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Construct a tensor representation of a manifold point.

        In case of regular manifolds this will return the same tensor. However, for e.g. Product manifold
        this function will pack all non-batch dimensions.

        Parameters
        ----------
        tensors : List[torch.Tensor]

        Returns
        -------
        torch.Tensor
        """
        if len(tensors) != 1:
            raise ValueError("Only one tensor expected")
        return tensors[0]


class ScaledFunction(object):
    r"""
    Helper class to scale method calls properly if a Manifold is wrapped into :class:`Scaled`

    To implement mixed curvature manifolds, within a single Product manifold we need to
    track scales of each tangent space. Moreover, if we eventually learn scalings, for some manifolds (e.g Sphere)
    we either should also change representations of points rescaling vectors in the ambient space, or change operations
    defined on these points. It is more practical to change operations, rather than points because this makes
    implementation much more easier.

    Examples
    --------
    Consider an arbitrary Riemannian Manifold.
    If we change the scale of charts on the manifold, distances would change as well. Say, we change
    distances at a constant factor :math:`\lambda`. Then having a convention that the only part changed is distance,
    but not points. We come up with a changed metric tensor what influences such operations as
    :math:`\operatorname{Exp}`, :math:`\operatorname{Log}` and so on.

    Examples
    --------
    For expmap we downscale tangent vector :math:`\Rightarrow` scaling power is -1. But output remains untouched
    :math:`\Rightarrow` scaling power is 0.

    >>> import geoopt, numpy as np
    >>> scaling_info_expmap = ScalingInfo(0, u=-1)
    >>> # scaling_info_expmap = ScalingInfo(u=-1) would be also valid, nothing to do with results
    >>> manifold = geoopt.Sphere()
    >>> scaled_expmap = ScaledFunction(manifold.expmap, scaling_info_expmap)
    >>> point = torch.tensor([2 ** .5 / 2, 2 ** .5 / 2])  # point on radius 1 sphere
    >>> tangent = manifold.proju(point, torch.randn(2))  # some random tangent there
    >>> new_point = scaled_expmap(point, tangent, scaling=2) # radius 2 sphere, but canonical representation is radius 1
    >>> new_point_alternative = manifold.expmap(point, tangent / 2)
    >>> np.testing.assert_allclose(new_point, new_point_alternative)
    """
    __slots__ = ["fn", "sig", "scaling_info"]

    def __init__(self, fn, scaling_info: ScalingInfo):
        self.fn = fn
        self.scaling_info = scaling_info
        self.sig = inspect.signature(fn)

    def __call__(self, *args, scaling, **kwargs):
        kwargs = self.sig.bind(*args, **kwargs).arguments
        for k, power in self.scaling_info.kwargs.items():
            kwargs[k] = kwargs[k] * scaling ** power
        results = self.fn(**kwargs)
        if not self.scaling_info.results:
            # do nothing
            return results
        if isinstance(results, tuple):
            return tuple(
                (
                    self.rescale(res, scaling, power)
                    for res, power in zip(results, self.scaling_info.results)
                )
            )
        else:
            power = self.scaling_info.results[0]
            return self.rescale(results, scaling, power)

    @staticmethod
    def rescale(value, scaling, power):
        if isinstance(power, torch.Tensor):
            return torch.where(power == 0, value, value * scaling ** power)
        else:
            return value * scaling ** power if power != 0 else value
