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

    def dist(self, x, y, *, keepdim=False):
        return math.dist(x, y, c=self.c, keepdim=keepdim)

    def egrad2rgrad(self, x, u):
        return math.egrad2rgrad(x, u, c=self.c)

    def retr(self, x, u):
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, c=self.c)

    def projx(self, x):
        return math.project(x, c=self.c)

    def proju(self, x, u):
        return u

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        return math.inner(x, u, v, c=self.c, keepdim=keepdim)

    def expmap(self, x, u):
        return math.project(math.expmap(x, u, c=self.c), c=self.c)

    def logmap(self, x, y):
        return math.logmap(x, y, c=self.c)

    def transp(self, x, y, v, *more):
        if not more:
            return math.parallel_transport(x, y, v, c=self.c)
        else:
            vecs = torch.stack((v,) + more, dim=0)
            transp = math.parallel_transport(x, y, vecs, c=self.c)
            return transp.unbind(0)

    def transp_follow_retr(self, x, u, v, *more):
        y = self.retr(x, u)
        return self.transp(x, y, v, *more)

    def transp_follow_expmap(self, x, u, v, *more):
        y = self.expmap(x, u)
        return self.transp(x, y, v, *more)

    def expmap_transp(self, x, u, v, *more):
        y = self.expmap(x, u)
        vs = self.transp(x, y, v, *more)
        return (y,) + make_tuple(vs)

    def retr_transp(self, x, u, v, *more):
        y = self.retr(x, u)
        vs = self.transp(x, y, v, *more)
        return (y,) + make_tuple(vs)


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
