from . import manifolds
from . import optim
from . import tensor
from . import samplers
from . import linalg

from .tensor import ManifoldParameter, ManifoldTensor
from .manifolds import (
    Stiefel,
    Euclidean,
    R,
    Sphere,
    SphereExact,
    PoincareBall,
    EuclideanStiefelExact,
)

__version__ = "0.0.1"
