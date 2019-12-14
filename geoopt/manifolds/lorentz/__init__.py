import torch as th
import torch.nn
from typing import Tuple, Optional
from . import math
import geoopt
from ..base import Manifold, ScalingInfo

__all__ = ["Lorentz"]


class Lorentz(Manifold):
    __doc__ = r"""{}
    """

    ndim = 1
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
        return True, None

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.egrad2rgrad(x, u, dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        approx = x + u
        return math.project(approx, dim=dim)

    def projx(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        return math.project(x, dim=dim)

    def proju():
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1
    ) -> torch.Tensor:
        res = math.expmap(x, u, dim=dim)
        if project:
            return math.project(res, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap(x, y, dim=dim)

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
        return math.egrad2rgrad(x, u, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, dim=-1):
        return math.parallel_transport(x, y, v, dim=dim)

    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, dim=dim)
        if project:
            return math.project(res, dim=dim)
        else:
            return res

    retr = expmap
