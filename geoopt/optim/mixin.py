from ..manifolds import Rn


class RiemannianOptimMixin(object):
    def __init__(self, manifold=Rn(), proj_x_every=1000):
        self.manifold = manifold
        self.proj_x_every = proj_x_every
