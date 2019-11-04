import inspect
import torch
import types
from typing import Union, Tuple, Optional
from geoopt.manifolds.base import Manifold, ScalingInfo
import geoopt.utils
import itertools
import functools

__all__ = ["Scaled"]


def rescale_value(value, scaling, power, manifold, attach):
    result = value * scaling ** power if power != 0 else value
    if attach:
        result = manifold.attach(result)
    return result


def apply_bound_args(
    f: callable, arguments: inspect.BoundArguments, signature: inspect.Signature
):
    args = list()
    kwargs = dict()
    for k, v in signature.parameters.items():
        if v.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            args.append(arguments.args[k])


def rescale(function, scaling_info):
    if scaling_info is ScalingInfo.NotCompatible:

        @functools.wraps(functools)
        def stub(self, *args, **kwargs):
            raise NotImplementedError(
                "Scaled version of '{}' is not available".format(function.__name__)
            )

        return stub
    signature = inspect.signature(function)

    @functools.wraps(function)
    def rescaled_function(self, *args, **kwargs):
        params = signature.bind(self.base, *args, **kwargs)
        params.apply_defaults()
        # TODO: varargs
        arguments = params.arguments
        for k, power in scaling_info.kwargs.items():
            arguments[k] = rescale_value(kwargs[k], self.scale, power, None, False)
        params = params.__class__(signature, arguments)
        results = function(*params.args, **params.kwargs)
        if not scaling_info.results and not scaling_info.attach:
            # do nothing, attach if needed
            return results if 0 not in scaling_info.attach else self.attach(results)
        wrapped_results = []
        is_tuple = isinstance(results, tuple)
        results = geoopt.utils.make_tuple(results)
        for i, (res, power) in enumerate(
            itertools.zip_longest(results, scaling_info.results, fillvalue=0)
        ):
            wrapped_results.append(
                rescale_value(res, self.scale, power, self, i in scaling_info.attach)
            )
        if not is_tuple:
            wrapped_results = wrapped_results[0]
        else:
            wrapped_results = results.__class__(wrapped_results)
        return wrapped_results

    return rescaled_function


class Scaled(Manifold):
    """
    Scaled manifold.

    Scales all the distances on tha manifold by a constant factor. Scaling may be learnable
    since the underlying representation is canonical.

    Examples
    --------
    Here is a simple example of radius 2 Sphere

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
            # register rescaled functions as bound methods of this particular instance
            unbound_method = getattr(self.base, method).__func__  # unbound method
            self.__setattr__(
                method, types.MethodType(rescale(unbound_method, scaling_info), self)
            )

    @property
    def scale(self) -> torch.Tensor:
        if self._scale is None:
            return self._log_scale.exp()
        else:
            return self._scale

    @property
    def log_scale(self) -> torch.Tensor:
        if self._log_scale is None:
            return self._scale.log()
        else:
            return self._log_scale

    # propagate all important stuff
    reversible = property(lambda self: self.base.reversible)
    ndim = property(lambda self: self.base.ndim)
    name = "Scaled"
    __scaling__ = property(lambda self: self.base.__scaling__)

    # Make AbstractMeta happy, to be fixed in __init__
    retr = NotImplemented
    expmap = NotImplemented

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as original:
            try:
                # propagate only public methods and attributes, ignore buffers, parameters, etc
                if isinstance(self.base, Scaled) and item in self._base_attributes:
                    return self.base.__getattr__(item)
                else:
                    return self.base.__getattribute__(item)
            except AttributeError as e:
                raise original from e

    @property
    def _base_attributes(self):
        if isinstance(self.base, Scaled):
            return self.base._base_attributes
        else:
            base_attribures = set(dir(self.base.__class__))
            base_attribures |= set(self.base.__dict__.keys())
            return base_attribures

    def __dir__(self):
        return list(set(super().__dir__()) | self.base_attribures)

    def __repr__(self):
        extra = self.base.extra_repr()
        if extra:
            return self.name + "({})({}) manifold".format(self.base.name, extra)
        else:
            return self.name + "({}) manifold".format(self.base.name)

    def _check_shape(self, shape: Tuple[int], name: str) -> Tuple[bool, Optional[str]]:
        return self.base._check_shape(shape, name)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return self.base._check_point_on_manifold(x, atol=atol, rtol=rtol)

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return self.base._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)

    # stuff that should remain the same but we need to override it
    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        **kwargs
    ) -> torch.Tensor:
        return self.base.inner(x, u, v, keepdim=keepdim, **kwargs)

    def norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, **kwargs
    ) -> torch.Tensor:
        return self.base.norm(x, u, keepdim=keepdim, **kwargs)

    def proju(self, x: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base.proju(x, u, **kwargs)

    def projx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.attach(self.base.projx(x, **kwargs))

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base.egrad2rgrad(x, u, **kwargs)

    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.base.transp(x, y, v, **kwargs)

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        return self.attach(
            self.base.random(*size, dtype=dtype, device=device, **kwargs)
        )

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42
    ) -> torch.Tensor:
        return self.attach(
            self.base.origin(*size, dtype=dtype, device=device, seed=seed)
        )
