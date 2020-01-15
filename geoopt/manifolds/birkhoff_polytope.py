import torch

from .base import Manifold
from .. import linalg
import torch.jit
from ..utils import make_tuple

__all__ = ["BirkhoffPolytope"]


class BirkhoffPolytope(Manifold):
    r"""
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

    @inproceedings{birdal2019probabilistic,
    title={Probabilistic Permutation Synchronization using the Riemannian Structure of the Birkhoff Polytope},
    author={Birdal, Tolga and Simsekli, Umut},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={11105--11116},
    year={2019}
    }

    This implementation is by Tolga Birdal and Haowen Deng.
    """

    name = "BirkhoffPolytope"

    ndim = 2

    def __init__(self, max_iter=100, tol=1e-5, eps=1e-12):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def _check_shape(self, shape, name):
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason

        shape_is_ok = shape[-1] == shape[-2]

        if not shape_is_ok:
            return (
                False,
                "`{}` should have shape[-1] == shape[-2], got {} != {}".format(
                    name, shape[-1], shape[-2]
                ),
            )

        return True, None

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        row_sum = x.sum(dim=-1)
        col_sum = x.sum(dim=-2)
        row_ok = torch.allclose(
            row_sum, row_sum.new((1,)).fill_(1), atol=atol, rtol=rtol
        )
        col_ok = torch.allclose(
            col_sum, col_sum.new((1,)).fill_(1), atol=atol, rtol=rtol
        )
        if row_ok and col_ok:
            return True, None
        else:
            return (
                False,
                "illegal doubly stochastic matrix with atol={}, rtol={}".format(
                    atol, rtol
                ),
            )

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x):
        return proj_doubly_stochastic(
            x=x, max_iter=self.max_iter, eps=self.eps, tol=self.tol
        )

    def proju(self, x, u):
        # takes batch data
        # batch_size, n, _ = x.shape
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])
        batch_size, n = x.shape[0:2]

        e = torch.ones(batch_size, n, 1, dtype=x.dtype)
        I = torch.unsqueeze(torch.eye(x.shape[-1], dtype=x.dtype), 0).repeat(
            batch_size, 1, 1
        )

        mu = x * u

        A = linalg.block_matrix([[I, x], [torch.transpose(x, 1, 2), I]])

        B = A[:, :, 1:]
        b = torch.cat(
            [
                torch.sum(mu, dim=2, keepdim=True),
                torch.transpose(torch.sum(mu, dim=1, keepdim=True), 1, 2),
            ],
            dim=1,
        )

        zeta, _ = torch.solve(
            B.transpose(1, 2) @ (b - A[:, :, 0:1]), B.transpose(1, 2) @ B
        )
        alpha = torch.cat(
            [torch.ones(batch_size, 1, 1, dtype=x.dtype), zeta[:, 0 : n - 1]], dim=1
        )
        beta = zeta[:, n - 1 : 2 * n - 1]
        rgrad = mu - (alpha @ e.transpose(1, 2) + e @ beta.transpose(1, 2)) * x

        rgrad = rgrad.reshape(x_shape)
        return rgrad

    egrad2rgrad = proju

    def retr(self, x, u):
        k = u / x
        y = x * torch.exp(k)
        y = self.projx(y)
        y = torch.max(y, y.new(1).fill_(1e-12))
        return y

    expmap = retr

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        n = x.shape[-1]
        return torch.sum(u * v / x, dim=(-1, -2), keepdim=keepdim) / n

    def transp(self, x, y, v):
        return self.proju(y, v)

    def retr_transp(self, x, u, v):
        y = self.retr(x, u)
        vs = self.transp(x, y, v)
        return (y,) + make_tuple(vs)

    expmap_transp = retr_transp

    def transp_follow_retr(self, x, u, v):
        y = self.retr(x, u)
        return self.transp(x, y, v)

    def transp_follow_expmap(self, x, u, v):
        y = self.expmap(x, u)
        return self.transp(x, y, v)


@torch.jit.script
def proj_doubly_stochastic(
    x, max_iter: int = 300, eps: float = 1e-5, tol: float = 1e-3
):
    iter = 0
    c = 1.0 / (x.sum(dim=-2, keepdim=True) + eps)
    r = 1.0 / ((x @ c.transpose(-1, -2)) + eps)
    while iter < max_iter:
        iter += 1
        cinv = torch.matmul(r.transpose(-1, -2), x)
        if torch.max(torch.abs(cinv * c - 1)) <= tol:
            break
        c = 1.0 / (cinv + eps)
        r = 1.0 / ((x @ c.transpose(-1, -2)) + eps)
    return x * (r @ c)
