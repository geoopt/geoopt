from ..base import Manifold


class Poincare(Manifold):
    ndim = 1
    reversible = True
    _default_order = 1
    name = "Poincare ball"
