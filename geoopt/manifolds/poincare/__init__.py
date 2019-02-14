from . import math as mobius_math
from ..base import Manifold

__all__ = ["PoincareBall"]


class PoincareBall(Manifold):
    """
    Poincare ball model, see more in :doc:`/extended/poincare`
    """

    ndim = 1
    reversible = False
    _default_order = 1
    name = "Poincare ball"
