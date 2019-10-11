import torch.nn
from . import math
from ...tensor import ManifoldTensor
from ...utils import size2shape, broadcast_shapes
from ..base import Manifold, ScalingInfo

__all__ = ["PoincareBall", "PoincareBallExact"]

_poincare_ball_doc = r"""
    Poincare ball model, see more in :doc:`/extended/poincare`.

    Parameters
    ----------
    c : float|tensor
        ball negative curvature

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


# noinspection PyMethodOverriding
class PoincareBall(Manifold):
    __doc__ = r"""{}

    See Also
    --------
    :class:`PoincareBallExact`
    """.format(
        _poincare_ball_doc
    )

    ndim = 1
    reversible = False
    name = "Poincare ball"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, c=1.0):
        super().__init__()
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.get_default_dtype()))

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        px = math.project(x, c=self.c)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
        return True, None

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1):
        return math.dist(x, y, c=self.c, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1):
        return math.egrad2rgrad(x, u, c=self.c, dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1):
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, c=self.c, dim=dim)

    def projx(self, x: torch.Tensor, dim=-1):
        return math.project(x, c=self.c, dim=dim)

    def proju(self, x: torch.Tensor, u: torch.Tensor):
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1
    ):
        if v is None:
            v = u
        return math.inner(x, u, v, c=self.c, keepdim=keepdim, dim=dim)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1):
        return math.norm(x, u, keepdim=keepdim, dim=dim)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1):
        res = math.expmap(x, u, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1):
        return math.logmap(x, y, c=self.c, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, dim=-1):
        return math.parallel_transport(x, y, v, c=self.c, dim=dim)

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1
    ):
        y = self.retr(x, u, dim=dim)
        return self.transp(x, y, v, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1, project=True
    ):
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1, project=True
    ):
        y = self.expmap(x, u, dim=dim, project=project)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1):
        y = self.retr(x, u, dim=dim)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True):
        res = math.mobius_add(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_sub(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True):
        res = math.mobius_sub(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_coadd(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True):
        res = math.mobius_coadd(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_cosub(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True):
        res = math.mobius_coadd(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_scalar_mul(
        self, r: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ):
        res = math.mobius_scalar_mul(r, x, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_pointwise_mul(
        self, w: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ):
        res = math.mobius_pointwise_mul(w, x, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True):
        res = math.mobius_matvec(m, x, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def geodesic(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1):
        return math.geodesic(t, x, y, c=self.c, dim=dim)

    @__scaling__(ScalingInfo(t=-1))
    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True
    ):
        res = math.geodesic_unit(t, x, u, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def lambda_x(self, x: torch.Tensor, *, dim=-1, keepdim=False):
        return math.lambda_x(x, c=self.c, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False):
        return math.dist0(x, c=self.c, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, dim=-1, project=True):
        res = math.expmap0(u, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(1))
    def logmap0(self, x: torch.Tensor, *, dim=-1):
        return math.logmap0(x, c=self.c, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1):
        return math.parallel_transport0(y, u, c=self.c, dim=dim)

    def transp0back(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1):
        return math.parallel_transport0back(y, u, c=self.c, dim=dim)

    def gyration(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, dim=-1):
        return math.gyration(x, y, z, c=self.c, dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist2plane(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        *,
        dim=-1,
        keepdim=False,
        signed=False
    ):
        return math.dist2plane(
            x, p, a, dim=dim, c=self.c, keepdim=keepdim, signed=signed
        )

    def mobius_fn_apply(
        self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs
    ):
        res = math.mobius_fn_apply(fn, x, *args, c=self.c, dim=dim, **kwargs)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_fn_apply_chain(
        self, x: torch.Tensor, *fns: callable, project=True, dim=-1
    ):
        res = math.mobius_fn_apply_chain(x, *fns, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def random_normal(self, *size, mean=0, std=1):
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

        Returns
        -------
        ManifoldTensor
            random point on the PoincareBall manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        tens = torch.randn(*size, device=self.c.device, dtype=self.c.dtype) * std + mean
        return ManifoldTensor(self.expmap0(tens), manifold=self)


class PoincareBallExact(PoincareBall):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`PoincareBall`
    """.format(
        _poincare_ball_doc
    )

    reversible = True
    retr_transp = PoincareBall.expmap_transp
    transp_follow_retr = PoincareBall.transp_follow_expmap
    retr = PoincareBall.expmap

    def extra_repr(self):
        return "exact"
