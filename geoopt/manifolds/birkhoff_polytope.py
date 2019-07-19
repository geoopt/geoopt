import torch

from .base import Manifold
from .. import linalg
from ..utils import  make_tuple

__all__ = ["BirkhoffPolytope"]

_birkhoff_doc = r"""
    Manifold induced by the Doubly Stochastic matrices as described in 
    A. Douik and B. Hassibi, "Manifold Optimization Over the Set 
    of Doubly Stochastic Matrices: A Second-Order Geometry"
    ArXiv:1802.02628, 2018.
    Link to the paper: https://arxiv.org/abs/1802.02628.

    @Techreport{Douik2018Manifold,
       Title   = {Manifold Optimization Over the Set of Doubly Stochastic 
                  Matrices: {A} Second-Order Geometry},
       Author  = {Douik, A. and Hassibi, B.},
       Journal = {Arxiv preprint ArXiv:1802.02628},
       Year    = {2018}
    }
    
    Please also cite:
    Tolga Birdal, Umut Şimşekli,
    "Probabilistic Permutation Synchronization using the Riemannian Structure of the BirkhoffPolytope Polytope"
    IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019
    Link to the paper: https://arxiv.org/abs/1904.05814
"""


class BirkhoffPolytope(Manifold):
    __doc__ = r"""
    {}
    """.format(_birkhoff_doc)
    name = "BirkhoffPolytope"

    ## batch_size \times n \times n
    ndim = 3

    def __init__(self, maxiter=100, tol=1e-3, epsilon=1e-12):
        self.maxiter = maxiter
        self.tol = tol
        self.epsilon = epsilon

    def _check_shape(self, shape, name):
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason

        # dim_is_ok = len(shape) == self.ndim
        #
        # if not dim_is_ok:
        #     return False, "dimensions for `{}` not match".format(name)
        #
        # return True, None

    def _check_point_on_manifold(self, x, *,  atol=1e-5, rtol=1e-5):
        row_sum = x.sum(dim=-1)
        col_sum = x.sum(dim=-2)
        row_ok = torch.allclose(row_sum, row_sum.new((1,)).fill_(1), atol=atol, rtol=rtol)
        col_ok = torch.allclose(col_sum, col_sum.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if row_ok and col_ok:
            return True, None
        else:
            return False, "illegal doubly stochastic matrix with atol={}, rtol={}".format(atol, rtol)

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x):
        iter = 0
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])
        c = 1.0 / (torch.sum(x, dim=-2, keepdim=True) + self.epsilon)
        r = 1.0 / (torch.matmul(x, torch.transpose(c, -1, -2)) + self.epsilon)
        while iter < self.maxiter:
            iter += 1
            cinv = torch.matmul(r.permute(0, 2, 1), x)
            if torch.max(torch.abs(cinv * c - 1)) <= self.tol:
                break
            c = 1.0 / (cinv + self.epsilon)
            r = 1.0 / (torch.matmul(x, torch.Tensor.permute(c, 0, 2, 1)) + self.epsilon)
        x = x * torch.matmul(r, c)
        x = x.reshape(x_shape)
        return x

    def proju(self, x, u):
        # takes batch data
        # batch_size, n, _ = x.shape
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])
        batch_size, n = x.shape[0:2]

        e = torch.ones(batch_size, n, 1)
        I = torch.unsqueeze(torch.eye(x.shape[-1]), 0).repeat(batch_size, 1, 1)

        mu = x * u

        A = linalg.block_matrix([[I, x], [torch.transpose(x, 1, 2), I]])

        B = A[:, :, 1:]
        b = torch.cat(
            [
                torch.sum(mu, dim=2, keepdim=True),
                torch.transpose(torch.sum(mu, dim=1, keepdim=True), 1, 2)
            ],
            dim=1
        )

        inv_B = torch.stack([torch.pinverse(_) for _ in B])
        zeta = torch.matmul(inv_B, b - A[:, :, 0:1])
        alpha = torch.cat([torch.ones(batch_size, 1, 1), zeta[:, 0:n - 1]], dim=1)
        beta = zeta[:, n - 1: 2 * n - 1]
        rgrad = mu - (alpha @ e.transpose(1, 2) + e @ beta.transpose(1, 2)) * x

        rgrad = rgrad.reshape(x_shape)
        return rgrad

    def retr(self, x, u):
        K = u / x
        Y = x * torch.exp(K)
        Y = self.projx(Y)
        Y = torch.max(Y, Y.new(1).fill_(1e-12))
        return Y

    def inner(self, x, u, v=None, *, keepdim=False):
        batch_size = x.shape[0]
        return torch.sum(u * v / x, keepdim=keepdim) / batch_size

    def transp(self, x, y, v, *more):
        if not more:
            return self.proju(y, v)
        else:
            return tuple(self.proju(y, v_) for v_ in (v,) + more)

    def retr_transp(self, x, u, v, *more):
        y = self.retr(x, u)
        vs = self.transp(x, y, v, *more)
        return (y,) + make_tuple(vs)
