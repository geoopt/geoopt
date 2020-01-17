from .base import Manifold
from .euclidean import Euclidean
from .stiefel import Stiefel, EuclideanStiefel, CanonicalStiefel, EuclideanStiefelExact
from .sphere import Sphere, SphereExact
from .stereographic import (
    PoincareBall,
    PoincareBallExact,
    Stereographic,
    StereographicExact,
    SphereProjection,
    SphereProjectionExact,
)
from .product import ProductManifold
from . import stereographic
from . import scaled
from .scaled import Scaled
