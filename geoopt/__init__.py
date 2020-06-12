from . import manifolds
from . import optim
from . import tensor
from . import samplers
from . import linalg
from . import utils
from .utils import ismanifold

from .tensor import ManifoldParameter, ManifoldTensor
from .manifolds import (
    Manifold,
    Stiefel,
    EuclideanStiefelExact,
    CanonicalStiefel,
    EuclideanStiefel,
    Euclidean,
    Sphere,
    SphereExact,
    PoincareBall,
    PoincareBallExact,
    Stereographic,
    StereographicExact,
    SphereProjection,
    SphereProjectionExact,
    ProductManifold,
    Scaled,
    Lorentz,
    BirkhoffPolytope,
)

__version__ = "0.2.0"
