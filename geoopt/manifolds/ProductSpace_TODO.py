import torch
from . import math
from geoopt.tensor import ManifoldTensor
from geoopt.utils import make_tuple, size2shape
from geoopt.manifolds.base import Manifold
from geoopt_ext import UniversalExact

__all__ = ["ProductSpace", "ProductSpaceExact"]

_universal_doc = r"""
Product space manifold model, see more in :doc:`/extended/product_space`
"""


# noinspection PyMethodOverriding
class ProductSpace(Manifold):
    __doc__ = r"""{}

    See Also
    --------
    :class:`ProductSpaceExact`
    """.format(
        _universal_doc
    )

    ndim = 1
    reversible = False
    name = "Universal"

    def __init__(self,
                 signature,
                 float_precision=torch.float64):

        # initialize Manifold superclass
        super().__init__()

        # keep track of supplied parameters
        self.supplied_signature = signature
        self.float_precision = float_precision

        # create manifold modules
        self.factors = torch.nn.ModuleList()

        # initialize factors
        for dim, K in signature:
            factor = UniversalExact(c=-K,
                                    float_precision=torch.float64,
                                    keep_sign_fixed=False,
                                    min_abs_c=0.001)
            self.factors.append(factor)

        # keep track of dimensions
        self.dim_ranges = []
        offset = 0
        for dim, K in signature:
            self.dim_ranges.append((offset, offset + dim))
            offset = offset + dim

    def get_dim_ranges(self, factor_idx):
        start = self.dim_ranges[factor_idx][0]
        end = self.dim_ranges[factor_idx][1]
        return start, end

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5):
        for i, factor in enumerate(self.factors):
            start, end = self.get_dim_ranges(i)
            factor.check_point_on_manifold(x[...,start:end])
            # TODO: return result

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        return True, None

    def dist(self, x, y, *, keepdim=False, dim=-1):

        sum_of_sq_dist = 0

        for i, factor in enumerate(self.factors):
            start, end = self.get_dim_ranges(i)
            factor_sq_dist = \
                factor.dist(x[...,start:end],
                            y[...,start:end],
                            keepdim=keepdim,
                            dim=-1).pow(2)
            sum_of_sq_dist = sum_of_sq_dist + factor_sq_dist

        return sum_of_sq_dist.sqrt()

    def egrad2rgrad(self, x, u, *, dim=-1):
        for i, factor in enumerate(self.factors):
            start, end = self.get_dim_ranges(i)
            # TODO: what to do about *
            h = factor.egrad2rgrad(x[...,start:end], u[...,start:end], dim=-1)
        # TODO: combine the gradients

    def retr(self, x, u, *, dim=-1):
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, c=self.get_c(), dim=dim)

    def projx(self, x, dim=-1):
        return math.project(x, c=self.get_c(), dim=dim)

    def proju(self, x, u):
        return u

    def inner(self, x, u, v=None, *, keepdim=False, dim=-1):
        if v is None:
            v = u
        return math.inner(x, u, v, c=self.get_c(), keepdim=keepdim, dim=dim)

    def norm(self, x, u, *, keepdim=False, dim=-1):
        return math.norm(x, u, keepdim=keepdim, dim=dim)

    def expmap(self, x, u, *, project=True, dim=-1):
        res = math.expmap(x, u, c=self.get_c(), dim=dim)
        if project:
            return math.project(res, c=self.get_c(), dim=dim)
        else:
            return res

    def logmap(self, x, y, *, dim=-1):
        return math.logmap(x, y, c=self.get_c(), dim=dim)

    def transp(self, x, y, v, *more, dim=-1):
        if not more:
            return math.parallel_transport(x, y, v, c=self.get_c(), dim=dim)
        else:
            return tuple(
                math.parallel_transport(x, y, vec, c=self.get_c(), dim=dim)
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

    def norm_constrain(self, x):

        with torch.no_grad():
            for i, factor in enumerate(self.factors):

                def norm_constrain_factor(x):
                    x_norm = x.norm(p=2, dim=-1, keepdim=True)
                    x_cond = x_norm <= factor.R_max
                    x_norm_constrained = factor.R_max * (x / x_norm)
                    x = torch.where(x_cond, x, x_norm_constrained)
                    return x

                start = self.dim_ranges[i][0]
                end = self.dim_ranges[i][1]
                x[...,start:end] = norm_constrain_factor(x[...,start:end])


class ProductSpaceExact(ProductSpace):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this 
    retraction will be used in optimization.

    See Also
    --------
    :class:`Universal`
    """.format(
        _universal_doc
    )

    reversible = True
    retr_transp = ProductSpace.expmap_transp
    transp_follow_retr = ProductSpace.transp_follow_expmap
    retr = ProductSpace.expmap

    def extra_repr(self):
        return "exact"
