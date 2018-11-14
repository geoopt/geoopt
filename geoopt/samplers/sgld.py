import math
import numpy as np
import torch

from ..tensor import ManifoldParameter, ManifoldTensor
from ..manifolds import Rn
from ..optim.mixin import OptimMixin
from .base import Sampler


class SGLD(Sampler):
    """Stochastic Gradient Langevin Dynamics"""

    def __init__(self, params, epsilon=1e-3):
        defaults = dict(epsilon=epsilon)
        super().__init__(params, defaults)
    
    
    def step(self, closure):
        """Performs a single sampling step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the log probability.
        """
        logp = closure()
        logp.backward()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    epsilon = group['epsilon']

                    n = torch.randn_like(p).mul_(math.sqrt(epsilon))
                    r = 0.5 * epsilon * p.grad + n

                    p.data.add_(r)
                    p.grad.data.zero_()

        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())


class RSGLD(OptimMixin, SGLD):
    """Riemannian Stochastic Gradient Langevin Dynamics"""

    def step(self, closure):
        """Performs a single sampling step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the log probability.
        """
        logp = closure()
        logp.backward()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                    else:
                        manifold = Rn()

                    proju, retr = manifold.proju, manifold.retr
                    epsilon = group['epsilon']

                    n = torch.randn_like(p).mul_(math.sqrt(epsilon))
                    r = proju(p.data, 0.5 * epsilon * p.grad + n)

                    p.data.set_(retr(p.data, r, 1.))
                    p.grad.data.zero_()

        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())


    def stabilize(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    continue

                p.data.set_(p.manifold.projx(p.data))
