import torch.nn
from .manifolds import Euclidean
from .docutils import insert_docs

__all__ = ["ManifoldTensor", "ManifoldParameter"]


class ManifoldTensor(torch.Tensor):
    """Same as :class:`torch.Tensor` that has information about its manifold.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold`
        A manifold for the tensor, (default: :class:`geoopt.Euclidean`)
    """

    def __new__(cls, *args, manifold=Euclidean(), requires_grad=False, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor.__new__(cls, *args, **kwargs)
        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))
        with torch.no_grad():
            manifold.assert_check_point(data)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    def proj_(self):
        """
        Inplace projection to the manifold

        Returns
        -------
        tensor
            same instance
        """
        with torch.no_grad():
            self.set_(self.manifold.projx(self.data))
        return self

    @insert_docs(Euclidean.retr.__doc__, r"\s+x : .+\n.+", "")
    def retr(self, u, t=1.0, order=None):
        return self.manifold.retr(self, u=u, t=t, order=order)

    @insert_docs(Euclidean.expmap.__doc__, r"\s+x : .+\n.+", "")
    def expmap(self, u, t=1.0):
        return self.manifold.expmap(self, u=u, t=t)

    @insert_docs(Euclidean.inner.__doc__, r"\s+x : .+\n.+", "")
    def inner(self, u, v=None):
        return self.manifold.inner(self, u=u, v=v)

    @insert_docs(Euclidean.proju.__doc__, r"\s+x : .+\n.+", "")
    def proju(self, u):
        return self.manifold.proju(self, u)

    @insert_docs(Euclidean.transp.__doc__, r"\s+x : .+\n.+", "")
    def transp(self, v, *more, u=None, t=1.0, y=None, order=None):
        return self.manifold.transp(self, v, *more, u=u, t=t, y=y, order=order)

    @insert_docs(Euclidean.retr_transp.__doc__, r"\s+x : .+\n.+", "")
    def retr_transp(self, v, *more, u, t=1.0, order=None):
        return self.manifold.retr_transp(self, u, *more, u=v, t=t, order=order)

    @insert_docs(Euclidean.expmap_transp.__doc__, r"\s+x : .+\n.+", "")
    def expmap_transp(self, v, *more, u, t=1.0):
        return self.manifold.expmap_transp(self, u, *more, u=v, t=t)

    def dist(self, other, p=2):
        """
        Return euclidean  or geodesic distance between points on the manifold. Allows broadcasting

        Parameters
        ----------
        other : tensor
        p : str|int
            The norm to use. The default behaviour is not changed and is just euclidean distance.
            To compute geodesic distance, :attr:`p` should be set to ``"g"``

        Returns
        -------
        scalar


        """
        if p == "g":
            return self.manifold.dist(self, other)
        else:
            return super().dist(other)

    @insert_docs(Euclidean.logmap.__doc__, r"\s+x : .+\n.+", "")
    def logmap(self, y):
        return self.manifold.logmap(self, y)

    def __repr__(self):
        return "Tensor on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)

    def rand_(self):
        with torch.no_grad():
            self.manifold.rand_(self.data)
        return self

    # noinspection PyUnresolvedReferences
    def __reduce_ex__(self, proto):
        proto = (
            self.__class__,
            self.storage(),
            self.storage_offset(),
            self.size(),
            self.stride(),
            self.requires_grad,
            dict(),
        )
        return _rebuild_manifold_parameter, proto + (self.manifold,)


class ManifoldParameter(ManifoldTensor, torch.nn.Parameter):
    """Same as :class:`torch.nn.Parameter` that has information about its manifold.
    It should be used within :class:`torch.nn.Module` to be recognized
    in parameter collection.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold` (optional)
        A manifold for the tensor if ``data`` is not a :class:`geoopt.ManifoldTensor`
    """

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


def _rebuild_manifold_parameter(cls, *args):
    import torch._utils

    tensor = torch._utils._rebuild_tensor_v2(*args[:-1])
    return cls(tensor, manifold=args[-1], requires_grad=args[-3])
