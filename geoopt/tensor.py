import torch.nn
from .manifolds import Euclidean


__all__ = ["ManifoldTensor", "ManifoldParameter"]


class ManifoldTensor(torch.Tensor):
    """A regular tensor that has information about its manifold.
    It is a very tiny wrapper over regular tensor so that all API is the same
    """

    def __new__(cls, *args, manifold=Euclidean(), requires_grad=False, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor.__new__(cls, *args, **kwargs)
        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))
        manifold.assert_check_point(data.data)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    def proj_(self):
        self.data.set_(self.manifold.projx(self.data))
        return self

    def retr(self, u, t):
        return self.manifold.retr(self, u, t)

    def inner(self, u, v=None):
        return self.manifold.inner(self, u, v)

    def proju(self, u):
        return self.manifold.proju(self, u)

    def transp(self, u, t, v, *more):
        return self.manifold.transp(self, u, t, v, *more)

    def retr_transp(self, u, t, v, *more):
        return self.manifold.retr_transp(self, u, t, v, *more)

    def __repr__(self):
        return "Tensor on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)


class ManifoldParameter(ManifoldTensor, torch.nn.Parameter):
    def __new__(cls, data=None, manifold=None, requires_grad=True):
        if data is None:
            data = ManifoldTensor(manifold=manifold)
        elif not isinstance(data, ManifoldTensor):
            data = ManifoldTensor(data, manifold=manifold or Euclidean())
        else:
            if manifold is not None and data.manifold != manifold:
                raise ValueError(
                    "Manifolds do not match: {}, {}".format(data.manifold, manifold)
                )
        instance = ManifoldTensor._make_subclass(cls, data, requires_grad)
        instance.manifold = data.manifold
        return instance

    def __repr__(self):
        return "Parameter on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)

    def __reduce_ex__(self, proto):
        return ManifoldParameter, (super(ManifoldParameter, self), self.requires_grad)
