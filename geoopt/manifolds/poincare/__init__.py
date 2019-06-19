import torch.nn
from . import math
from ...utils import make_tuple
from ..base import Manifold

__all__ = ["PoincareBall", "PoincareBallExact"]

_poincare_ball_doc = r"""
    Poincare ball model, see more in :doc:`/extended/poincare`

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

    def __init__(self, c=1.0):
        super().__init__()
        self.register_buffer("c", torch.as_tensor(c))

    def _check_shape(self, x, name):
        ok = x.dim() > 0
        if not ok:
            reason = "'{}' on poincare ball requires more that zero dim".format(name)
        else:
            reason = None
        return ok, reason

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        px = math.project(x, c=self.c)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        return True, None

    def dist(self, x, y, *, keepdim=False, dim=-1):
        return math.dist(x, y, c=self.c, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x, u, *, dim=-1):
        return math.egrad2rgrad(x, u, c=self.c, dim=dim)

    def retr(self, x, u, *, dim=-1):
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, c=self.c, dim=dim)

    def projx(self, x, dim=-1):
        return math.project(x, c=self.c, dim=dim)

    def proju(self, x, u):
        return u

    def inner(self, x, u, v=None, *, keepdim=False, dim=-1):
        if v is None:
            v = u
        return math.inner(x, u, v, c=self.c, keepdim=keepdim, dim=dim)

    def norm(self, x, u, *, keepdim=False, dim=-1):
        return math.norm(x, u, keepdim=keepdim, dim=dim)

    def expmap(self, x, u, *, project=True, dim=-1):
        res = math.expmap(x, u, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def logmap(self, x, y, *, dim=-1):
        return math.logmap(x, y, c=self.c, dim=dim)

    def transp(self, x, y, v, *more, dim=-1):
        if not more:
            return math.parallel_transport(x, y, v, c=self.c, dim=dim)
        else:
            return tuple(
                math.parallel_transport(x, y, vec, c=self.c, dim=dim)
                for vec in (v, *more)
            )

    def transp_follow_retr(self, x, u, v, *more, dim=-1):
        y = self.retr(x, u, dim=dim)
        return self.transp(x, y, v, *more, dim=dim)

    def transp_follow_expmap(self, x, u, v, *more, dim=-1, project=True):
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, *more, dim=dim)

    def expmap_transp(self, x, u, v, *more, dim=-1, project=True):
        y = self.expmap(x, u, dim=dim, project=project)
        vs = self.transp(x, y, v, *more, dim=dim)
        return (y,) + make_tuple(vs)

    def retr_transp(self, x, u, v, *more, dim=-1):
        y = self.retr(x, u, dim=dim)
        vs = self.transp(x, y, v, *more, dim=dim)
        return (y,) + make_tuple(vs)

    def mobius_add(self, x, y, *, dim=-1, project=True):
        res = math.mobius_add(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_sub(self, x, y, *, dim=-1, project=True):
        res = math.mobius_sub(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_coadd(self, x, y, *, dim=-1, project=True):
        res = math.mobius_coadd(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_cosub(self, x, y, *, dim=-1, project=True):
        res = math.mobius_coadd(x, y, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_scalar_mul(self, r, x, *, dim=-1, project=True):
        res = math.mobius_scalar_mul(r, x, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_pointwise_mul(self, w, x, *, dim=-1, project=True):
        res = math.mobius_pointwise_mul(w, x, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_matvec(self, m, x, *, dim=-1, project=True):
        res = math.mobius_matvec(m, x, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def geodesic(self, t, x, y, *, dim=-1):
        return math.geodesic(t, x, y, c=self.c, dim=dim)

    def geodesic_unit(self, t, x, u, *, dim=-1, project=True):
        res = math.geodesic_unit(t, x, u, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def lambda_x(self, x, *, dim=-1, keepdim=False):
        return math.lambda_x(x, c=self.c, dim=dim, keepdim=keepdim)

    def dist0(self, x, *, dim=-1, keepdim=False):
        return math.dist0(x, c=self.c, dim=dim, keepdim=keepdim)

    def expmap0(self, u, *, dim=-1, project=True):
        res = math.expmap0(u, c=self.c, dim=dim)
        if project:
            return math.project(res, c=self.c, dim=dim)
        else:
            return res

    def logmap0(self, x, *, dim=-1):
        return math.logmap0(x, c=self.c, dim=dim)

    def transp0(self, y, u, *, dim=-1):
        return math.parallel_transport0(y, u, c=self.c, dim=dim)

    def transp0back(self, y, u, *, dim=-1):
        return math.parallel_transport0back(y, u, c=self.c, dim=dim)

    def gyration(self, x, y, z, *, dim=-1):
        return math.gyration(x, y, z, c=self.c, dim=dim)

    def dist2plane(self, x, p, a, *, dim=-1, keepdim=False, signed=False):
        return math.dist2plane(
            x, p, a, dim=dim, c=self.c, keepdim=keepdim, signed=signed
        )


class PoincareBallExact(PoincareBall):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization
    
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
