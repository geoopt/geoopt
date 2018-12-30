import math

import torch

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Euclidean
from geoopt.samplers.base import Sampler


__all__ = ["RSGLD"]


class RSGLD(Sampler):
    """Riemannian Stochastic Gradient Langevin Dynamics

    Parameters
    ----------
    params : iterable[tensor]
        iterables of tensors for which to perform sampling
    epsilon : float
        step size

    """

    def __init__(self, params, epsilon=1e-3):
        defaults = dict(epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure):
        """Performs a single sampling step.

        Arguments
        ---------
        closure: callable
            A closure that reevaluates the model
            and returns the log probability.
        """
        logp = closure()
        logp.backward()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                    else:
                        manifold = Euclidean()

                    proju, retr = manifold.proju, manifold.retr
                    epsilon = group["epsilon"]

                    n = torch.randn_like(p).mul_(math.sqrt(epsilon))
                    r = proju(p.data, 0.5 * epsilon * p.grad + n)

                    p.data.set_(retr(p.data, r, 1.0))
                    p.grad.data.zero_()

        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons
        """
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    continue

                p.data.set_(p.manifold.projx(p.data))
