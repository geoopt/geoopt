import inspect
import torch
from geoopt.manifolds.base import ScalingInfo, Manifold
import functools

__all__ = ["Scaled"]


class ScaledFunction(object):
    r"""
    Helper class to scale method calls properly if a Manifold is wrapped into :class:`Scaled`.

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

    For expmap we downscale tangent vector :math:`\Rightarrow` scaling power is -1. But output remains untouched
    :math:`\Rightarrow` scaling power is 0.

    >>> import geoopt, torch, numpy as np
    >>> scaling_info_expmap = ScalingInfo(u=-1)
    >>> manifold = geoopt.Sphere()
    >>> scaled_expmap = ScaledFunction(manifold.expmap, scaling_info_expmap)
    >>> point = torch.tensor([2 ** .5 / 2, 2 ** .5 / 2])  # point on radius 1 sphere
    >>> tangent = manifold.proju(point, torch.randn(2))  # some random tangent there
    >>> new_point = scaled_expmap(point, tangent, scaling=2) # radius 2 sphere, but canonical representation is radius 1
    >>> new_point_alternative = manifold.expmap(point, tangent / 2)
    >>> np.testing.assert_allclose(new_point, new_point_alternative)
    """

    __slots__ = ["__signature__", "__wrapped__", "fsig", "scaling_info"]

    def __init__(self, fn: callable, scaling_info: ScalingInfo):
        self.__wrapped__ = fn
        self.scaling_info = scaling_info
        self.fsig = inspect.signature(fn)

        # for correct and consistent behaviour and introspection
        # we need to specify signature attribute.
        old_args = tuple(self.fsig.parameters.values())
        # we use scaling and log, but actually we may rescale the existent scale function
        # this is to avoid collision
        old_args = tuple(
            filter(lambda arg: arg.name not in {"scaling", "log_scaling"}, old_args)
        )
        # variadic keywords case
        if old_args and old_args[-1].kind is inspect.Parameter.VAR_KEYWORD:
            variadic = (old_args[-1],)
            old_args = old_args[:-1]
        else:
            variadic = ()

        new_args = (
            inspect.Parameter(
                "scaling", kind=inspect.Parameter.KEYWORD_ONLY, default=None
            ),
            inspect.Parameter(
                "log_scaling", kind=inspect.Parameter.KEYWORD_ONLY, default=None
            ),
        )
        self.__signature__ = self.fsig.replace(
            parameters=old_args + new_args + variadic
        )

    def __call__(self, *args, scaling=None, log_scaling=None, **kwargs):
        assert (scaling is not None) ^ (log_scaling is not None)
        if log_scaling is not None:
            scaling = log_scaling.exp()
        kwargs = self.fsig.bind(*args, **kwargs).arguments
        for k, power in self.scaling_info.kwargs.items():
            kwargs[k] = self.rescale(kwargs[k], scaling, power)
        results = self.__wrapped__(**kwargs)
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
        return value * scaling ** power if power != 0 else value


class Scaled(Manifold):
    """
    Scaled manifold.

    Scales all the distances on tha manifold by a constant factor. Scaling may be learnable
    since the underlying representation is canonical.

    Examples
    --------
    Here is a simple example of

    >>> import geoopt, torch, numpy as np
    >>> sphere = geoopt.Sphere()
    >>> radius_2_sphere = Scaled(sphere, 2)
    >>> p1 = torch.tensor([-1., 0.])
    >>> p2 = torch.tensor([0., 1.])
    >>> np.testing.assert_allclose(sphere.dist(p1, p2), np.pi / 2)
    >>> np.testing.assert_allclose(radius_2_sphere.dist(p1, p2), np.pi)
    """

    def __init__(self, manifold: Manifold, scale=1.0, learnable=False):
        super().__init__()
        self.base = manifold
        self._scaled_functions = dict()
        scale = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        scale = scale.requires_grad_(False)
        if not learnable:
            self.register_buffer("_scale", scale)
            self.register_buffer("_log_scale", None)
        else:
            self.register_buffer("_scale", None)
            self.register_parameter("_log_scale", torch.nn.Parameter(scale.log()))
        # do not rebuild scaled functions very frequently, save them

        for method, scaling_info in self.base.__scaling__.items():
            bound_method = getattr(self.base, method)
            self._scaled_functions[method] = ScaledFunction(bound_method, scaling_info)

    @property
    def scale(self):
        if self._scale is None:
            return self._log_scale.exp()
        else:
            return self._scale

    @property
    def log_scale(self):
        if self._log_scale is None:
            return self._scale.log()
        else:
            return self._log_scale

    # propagate all important stuff
    reversible = property(lambda self: self.base.reversible)
    ndim = property(lambda self: self.base.ndim)
    name = "Scaled"
    __scaling__ = property(lambda self: self.base.__scaling__)

    def __getattr__(self, item):
        if item in self._scaled_functions:
            return functools.partial(
                self._scaled_functions[item],
                # for partial we need raw tensors
                scaling=self._scale,
                log_scaling=self._log_scale,
            )
        else:
            try:
                return super().__getattr__(item)
            except AttributeError as original:
                try:
                    return self.base.__getattribute__(item)
                except AttributeError as e:
                    raise original from e

    def __dir__(self):
        base_attribures = set(dir(self.base.__class__))
        base_attribures |= set(self.base.__dict__.keys())
        return list(set(super().__dir__()) | base_attribures)

    def __repr__(self):
        extra = self.base.extra_repr()
        if extra:
            return self.name + "({})({}) manifold".format(self.base.name, extra)
        else:
            return self.name + "({}) manifold".format(self.base.name)

    def _check_shape(self, shape, name):
        return self.base._check_shape(shape, name)

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        return self.base._check_point_on_manifold(x, atol=atol, rtol=rtol)

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        return self.base._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)

    def retr(self, x: torch.Tensor, u: torch.Tensor, **kwargs):
        return self._scaled_functions["retr"](x, u, scaling=self.scale, **kwargs)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        **kwargs
    ):
        return self.base.inner(x, u, v, keepdim=keepdim, **kwargs)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, **kwargs):
        return self.base.norm(x, u, keepdim=keepdim, **kwargs)

    def proju(self, x: torch.Tensor, u: torch.Tensor, **kwargs):
        return self.base.proju(x, u, **kwargs)

    def projx(self, x: torch.Tensor, **kwargs):
        return self.base.projx(x, **kwargs)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        return self._scaled_functions["logmap"](x, y, scaling=self.scale, **kwargs)

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, **kwargs):
        return self._scaled_functions["dist"](
            x, y, keepdim=keepdim, scaling=self.scale, **kwargs
        )

    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, **kwargs):
        return self._scaled_functions["dist2"](
            x, y, keepdim=keepdim, scaling=self.scale, **kwargs
        )

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, **kwargs):
        return self.base.egrad2rgrad(x, u, **kwargs)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, **kwargs):
        return self._scaled_functions["expmap"](x, u, scaling=self.scale, **kwargs)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, **kwargs):
        return self.base.transp(x, y, v, **kwargs)

    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, **kwargs):
        return self._scaled_functions["retr_transp"](
            x, u, v, scaling=self.scale, **kwargs
        )

    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, **kwargs
    ):
        return self._scaled_functions["expmap_transp"](
            x, u, v, scaling=self.scale, **kwargs
        )

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, **kwargs
    ):
        return self._scaled_functions["transp_follow_expmap"](
            x, u, v, scaling=self.scale, **kwargs
        )

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, **kwargs
    ):
        return self._scaled_functions["transp_follow_retr"](
            x, u, v, scaling=self.scale, **kwargs
        )
