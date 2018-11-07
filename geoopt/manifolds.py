import abc
import torch
from . import util


class Manifold(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def retr(self, x, u, t):
        raise NotImplementedError

    @abc.abstractmethod
    def transp(self, x, u, v, t):
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, x, u, v):
        raise NotImplementedError

    def norm(self, x, u):
        return self.inner(x, u, u)

    @abc.abstractmethod
    def proju(self, x, u):
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x):
        raise NotImplementedError


class Rn(Manifold):
    def retr(self, x, u, t):
        return x + t * u

    def inner(self, x, u, v):
        return (u * v).sum(-1)

    def proju(self, x, u):
        return u

    def projx(self, x):
        return x

    def transp(self, x, u, v, t):
        return v


class Stiefel(Manifold):
    def amat(self, x, u, project=True):
        if project:
            u = self.proju(x, u)
        return u @ x.transpose(-1, -2) - x @ u.transpose(-1, -2)

    def proju(self, x, u):
        p = -0.5 * x @ x.transpose(-1, -2)
        p[..., range(x.shape[-2]), range(x.shape[-2])] += 1
        return p @ u

    def projx(self, x):
        U, d, V = util.svd(x)
        return torch.einsum("...ik,...k,...jk->...ij", [U, torch.ones_like(d), V])

    def retr(self, x, u, t):
        a = self.amat(x, u, project=False)
        rhs = x + t / 2 * a @ x
        lhs = -t / 2 * a
        lhs[..., range(a.shape[-2]), range(x.shape[-2])] += 1
        qx, _ = torch.gesv(rhs, lhs)
        return qx

    def inner(self, x, u, v):
        return (u * v).sum([-1, -2])

    def transp(self, x, u, v, t):
        a = self.amat(x, u, project=False)
        rhs = v + t / 2 * a @ v
        lhs = -t / 2 * a
        lhs[..., range(a.shape[-2]), range(x.shape[-2])] += 1
        qv, _ = torch.gesv(rhs, lhs)
        return qv
