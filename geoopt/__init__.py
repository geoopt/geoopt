from . import manifolds
from . import optim
from . import tensor
from . import samplers
from . import linalg

from .tensor import ManifoldParameter, ManifoldTensor
from .manifolds import (
    Stiefel,
    Euclidean,
    Sphere,
    SphereSubspaceIntersection,
    SphereSubspaceComplementIntersection,
)

__version__ = "0.0.1"
