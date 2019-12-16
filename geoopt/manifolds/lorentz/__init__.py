import torch as th
import torch.nn
from typing import Tuple, Optional
from . import math
import geoopt
from ..base import Manifold, ScalingInfo
from ...utils import size2shape, broadcast_shapes

__all__ = ["Lorentz"]

_lorentz_ball_doc = r"""
    Hyperboloid model

    Parameters
    ----------
    k : float|tensor
        manifold negative curvature
"""

class Lorentz(Manifold):
    __doc__ = r"""{}
    """

    ndim = 1
    reversible = False
    name = "Hyperboloid"

    def __init__(self, k=1.0):
        super().__init__()
        self.register_buffer("k", torch.as_tensor(k, dtype=torch.get_default_dtype()))

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        dn = x.size(0)
        x = x ** 2
        quad_form = -x[0] + x[1:].sum()
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        if not ok:
            reason = f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        inner_ = math.inner(u, x)
        ok = torch.allclose(inner_, torch.zeros(1), atol=atol, rtol=rtol)
        if not ok:
            reason = f"Minkowski inner produt is not equal to zero"
        else:
            reason = None
        return ok, reason

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, dim=dim)

    def projx(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, dim=-1) -> torch.Tensor:
        v = math.project_u(x, v, k=self.k, dim=dim)
        return v

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1
    ) -> torch.Tensor:
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def expmap0(self, u: torch.Tensor, *, project=True, dim=-1) -> torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        return math.inner(x, u, v)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, dim=-1):
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1, project=True
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def random_uniform(self, *size, dtype=None, device=None) -> "geoopt.ManifoldTensor":
        """
        Uniform sampling in hyperbolic space

        Parameters

        ----------
        size : shape
            the desired shape
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        elems = torch.rand(*size)
        r = math._arcosh(1. + elems * (th.cosh(self.k) - 1.))
        return r

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            zero point on the manifold
        """
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )


    retr = expmap
