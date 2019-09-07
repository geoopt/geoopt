import torch.nn
from typing import Tuple, Union
import operator
import functools
from .base import Manifold
from ..utils import size2shape


__all__ = ["ProductManifold"]


def _shape2size(shape):
    return functools.reduce(operator.mul, shape, 1)


class ProductManifold(Manifold):
    """Product Manifold."""

    ndim = 1

    def __init__(
        self, *manifolds_with_shape: Tuple[Manifold, Union[Tuple[int, ...], int]]
    ):
        if len(manifolds_with_shape) < 1:
            raise ValueError(
                "There should be at least one manifold in a product manifold"
            )
        super().__init__()
        self.shapes = []
        self.slices = []
        name_parts = []
        manifolds = []
        pos0 = 0
        for i, (manifold, shape) in enumerate(manifolds_with_shape):
            shape = size2shape(shape)
            ok, reason = manifold._check_shape(shape, str("{}'th shape".format(i)))
            if not ok:
                raise ValueError(reason)
            name_parts.append(manifold.name)
            manifolds.append(manifold)
            self.shapes.append(shape)
            pos1 = pos0 + _shape2size(shape)
            self.slices.append(slice(pos0, pos1))
            pos0 = pos1
        self.name = "x".join(["({})".format(name) for name in name_parts])
        self.n_elements = pos0
        self.n_manifolds = len(manifolds)
        self.manifolds = torch.nn.ModuleList(manifolds)

    def take_submanifold_value(self, x: torch.Tensor, i: int, reshape=True):
        """
        Take i'th slice of the ambient tensor and possibly reshape.

        Parameters
        ----------
        x : tensor
            Ambient tensor
        i : int
            submanifold index
        reshape : bool
            reshape the slice?

        Returns
        -------
        tensor
        """
        slc = self.slices[i]
        part = x.narrow(-1, slc.start, slc.stop - slc.start)
        if reshape:
            part = part.view(*part.shape[:-1], *self.shapes[i])
        return part

    def _check_shape(self, shape, name):
        ok = shape[-1] == self.n_elements
        if not ok:
            return (
                ok,
                "The last dimension should be equal to {}, but got {}".format(
                    self.n_elements, shape[-1]
                ),
            )
        return ok, None

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        ok, reason = True, None
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            ok, reason = manifold.check_point_on_manifold(
                point, atol=atol, rtol=rtol, explain=True
            )
            if not ok:
                break
        return ok, reason

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        ok, reason = True, None
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            ok, reason = manifold.check_vector_on_tangent(
                point, atol=atol, rtol=rtol, explain=True
            )
            if not ok:
                break
        return ok, reason

    def inner(self, x, u, v=None, *, keepdim=False):
        target_batch_dim = max(x.dim(), u.dim())
        if v is not None:
            target_batch_dim = max(target_batch_dim, v.dim())
        target_batch_dim -= 1
        products = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            u_vec = self.take_submanifold_value(u, i)
            if v is not None:
                v_vec = self.take_submanifold_value(v, i)
            else:
                v_vec = None
            inner = manifold.inner(point, u_vec, v_vec)
            inner = inner.view(*inner.shape[:target_batch_dim], -1).sum(-1)
            products.append(inner)
        result = sum(products)
        if keepdim:
            result = torch.unsqueeze(result, -1)
        return result

    def projx(self, x):
        ...

    def proju(self, x, u):
        ...

    def expmap(self, x, u):
        ...

    def retr(self, x, u):
        ...

    def transp(self, x, y, v):
        ...

    def logmap(self, x, y):
        ...

    def dist(self, x, y, *, keepdim=False):
        ...

    def egrad2rgrad(self, x, u):
        ...

    def as_point(self, tensor: torch.Tensor):
        parts = []
        for i in range(self.n_manifolds):
            part = self.take_submanifold_value(tensor, i)
            parts.append(part)
        return tuple(parts)

    def as_tensor(self, *parts: torch.Tensor):
        flattened = []
        for i, part in enumerate(parts):
            shape = self.shapes[i]
            if len(shape) > 0:
                if part.shape[-len(shape) :] != shape:
                    raise ValueError(
                        "last shape dimension does not seem to be valid. {} required, but got {}".format(
                            part.shape[-len(shape) :], shape
                        )
                    )
                new_shape = (*part.shape[: -len(shape)], -1)
            else:
                new_shape = (*part.shape, -1)
            flattened.append(part.view(new_shape))
        return torch.cat(flattened, -1)
