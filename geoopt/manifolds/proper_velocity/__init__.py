from typing import List, Optional, Tuple

import torch

import geoopt

from ...utils import size2shape
from ..base import Manifold, ScalingInfo
from . import math

__all__ = ["ProperVelocity"]

_proper_velocity_doc = r"""
    Proper Velocity (PV) model of hyperbolic space.

    Points are arbitrary vectors :math:`x \in \mathbb{R}^n`: the space-like coordinates of the
    hyperboloid :math:`-x_0^2 + \|x\|^2 = -k`. The model therefore has constant curvature
    :math:`-1/k` (the same convention as :class:`geoopt.Lorentz`), but no boundary and no constraint,
    and points have the same shape as Euclidean features.

    Parameters
    ----------
    k : float|tensor
        manifold negative curvature, the sectional curvature is :math:`-1/k`
    learnable : bool
        make ``k`` a learnable parameter

    Notes
    -----
    See Chen et al., "Proper Velocity Neural Networks", ICLR 2026.
"""


class ProperVelocity(Manifold):
    __doc__ = r"""{}

    Examples
    --------
    >>> import geoopt, torch
    >>> pv = geoopt.ProperVelocity(k=1.0)
    >>> x = torch.tensor([3.0, 4.0])
    >>> torch.allclose(pv.dist0(x), torch.asinh(torch.tensor(5.0)))
    True
    """.format(_proper_velocity_doc)

    ndim = 1
    reversible = True
    name = "ProperVelocity"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, k=1.0, learnable=False):
        super().__init__()
        k = torch.as_tensor(k)
        if not torch.is_floating_point(k):
            k = k.to(torch.get_default_dtype())
        if (k <= 0).any():
            raise ValueError("k must be positive")
        self.k = torch.nn.Parameter(k, requires_grad=learnable)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        ok = bool(torch.isfinite(x).all())
        reason = None if ok else "'x' contains non-finite values"
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        ok = bool(torch.isfinite(u).all())
        reason = None if ok else "'u' contains non-finite values"
        return ok, reason

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return u

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        if v is None:
            v = u
        return math.inner(x, u, v, k=self.k, keepdim=keepdim, dim=dim)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.norm(x, u, k=self.k, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.expmap(x, u, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.expmap0(u, k=self.k, dim=dim)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(1))
    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(y, k=self.k, dim=dim)

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return math.dist0(x, k=self.k, keepdim=keepdim, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim)
        return self.transp(x, y, v, dim=dim)

    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.expmap(x, u, dim=dim)
        return y, self.transp(x, y, v, dim=dim)

    @__scaling__(ScalingInfo.NotCompatible)
    def gyroadd(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.gyroadd(x, y, k=self.k, dim=dim)

    def gyroinv(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.gyroinv(x, dim=dim)

    @__scaling__(ScalingInfo.NotCompatible)
    def gyroscalar(self, r: torch.Tensor, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.gyroscalar(r, x, k=self.k, dim=dim)

    @__scaling__(ScalingInfo.NotCompatible)
    def gyration(
        self, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.gyration(a, b, x, k=self.k, dim=dim)

    def geodesic(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.geodesic(t, x, y, k=self.k, dim=dim)

    def frechet_mean(
        self,
        x: torch.Tensor,
        *,
        reducedim: Optional[List[int]] = None,
        dim=-1,
        keepdim=False,
        max_iter=100,
        tol=1e-6,
    ) -> torch.Tensor:
        return math.frechet_mean(
            x,
            k=self.k,
            reducedim=reducedim,
            dim=dim,
            keepdim=keepdim,
            max_iter=max_iter,
            tol=tol,
        )

    @__scaling__(ScalingInfo(2))
    def frechet_variance(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        *,
        reducedim: Optional[List[int]] = None,
        dim=-1,
        keepdim=False,
    ) -> torch.Tensor:
        return math.frechet_variance(
            x, mean, k=self.k, reducedim=reducedim, dim=dim, keepdim=keepdim
        )

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on the PV manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        size = size2shape(*size)
        self._assert_check_shape(size, "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the manifold `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the manifold `dtype`, set the `dtype` argument to None"
            )
        tens = (
            torch.randn(size, device=self.k.device, dtype=self.k.dtype) * std / size[-1] ** 0.5
            + mean
        )
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    random = random_normal

    def origin(self, *size, dtype=None, device=None, seed=42) -> "geoopt.ManifoldTensor":
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
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device
        return geoopt.ManifoldTensor(torch.zeros(*size, dtype=dtype, device=device), manifold=self)

    def extra_repr(self):
        return "k={}".format(self.k.item() if self.k.numel() == 1 else self.k)

    retr = expmap
