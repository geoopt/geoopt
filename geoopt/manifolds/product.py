import torch.nn
from typing import Tuple, Union
from .base import Manifold
from ..utils import size2shape


class ProductManifold(Manifold):
    def __init__(
        self, *manifolds_with_shape: Tuple[Manifold, Union[Tuple[int, ...], int]]
    ):
        super().__init__()
        manifolds = []
        self.shapes = []
        for i, (manifold, shape) in enumerate(manifolds_with_shape):
            ok, reason = manifold._check_shape(shape, str("{}'th shape".format(i)))
            if not ok:
                raise ValueError(reason)
            manifolds.append(manifold)
            self.shapes.append(size2shape(shape))
        self.manifolds = torch.nn.ModuleList(manifolds)
