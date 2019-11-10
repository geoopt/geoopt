from . import manifolds
from . import optim
from . import tensor
from . import samplers
from . import linalg

from .tensor import ManifoldParameter, ManifoldTensor
from .manifolds import (
    Stiefel,
    EuclideanStiefelExact,
    CanonicalStiefel,
    EuclideanStiefel,
    Euclidean,
    Sphere,
    SphereExact,
    PoincareBall,
    PoincareBallExact,
    ProductManifold,
    Scaled,
)
from .utils import ismanifold

__version__ = "0.1.0"
