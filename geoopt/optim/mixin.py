from ..manifolds import Rn


class RiemannianOptimMixin(object):
    def __init__(self, manifold=Rn(), stabilize=1000):
        self.defaults = dict(manifold=manifold, stabilize=stabilize)

    @property
    def defaults(self):
        return self._defaults

    @defaults.setter
    def defaults(self, value):
        if not hasattr(self, '_defaults'):
            self._defaults = value
        else:
            self._defaults.uprate(value)
