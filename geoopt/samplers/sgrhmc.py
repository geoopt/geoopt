import math

import torch

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Euclidean
from geoopt.samplers.base import Sampler


__all__ = ["SGRHMC"]


class SGRHMC(Sampler):
    r"""Stochastic Gradient Riemannian Hamiltonian Monte-Carlo

    Parameters
    ----------
    params : iterable
        iterables of tensors for which to perform sampling
    epsilon : float
        step size
    n_steps : int
        number of leapfrog steps
    alpha : float
        :math:`(1 - alpha)` -- momentum term
    """

    def __init__(self, params, epsilon=1e-3, n_steps=1, alpha=0.1):
        defaults = dict(epsilon=epsilon, alpha=alpha)
        super().__init__(params, defaults)
        self.n_steps = n_steps

    def step(self, closure):
        """Performs a single sampling step.

        Arguments
        ---------
        closure: callable
            A closure that reevaluates the model
            and returns the log probability.
        """
        H_old = 0.0
        H_new = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                if "v" not in state:
                    state["v"] = torch.zeros_like(p)

                epsilon = group["epsilon"]
                v = state["v"]
                v.normal_().mul_(epsilon)

                r = v / epsilon
                H_old += 0.5 * (r * r).sum().item()

        for i in range(self.n_steps + 1):
            logp = closure()
            logp.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                            manifold = p.manifold
                        else:
                            manifold = Euclidean()

                        proju = manifold.proju
                        retr_transp = manifold.retr_transp

                        epsilon, alpha = group["epsilon"], group["alpha"]

                        v = self.state[p]["v"]

                        p_, v_ = retr_transp(p, v, 1.0, v)
                        p.set_(p_)
                        v.set_(v_)

                        n = proju(p, torch.randn_like(v))
                        v.mul_(1 - alpha).add_(epsilon * p.grad).add_(
                            math.sqrt(2 * alpha * epsilon) * n
                        )
                        p.grad.zero_()

                        r = v / epsilon
                        H_new += 0.5 * (r * r).sum().item()

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

                manifold = p.manifold
                v = self.state[p]["v"]

                p.set_(manifold.projx(p))
                v.set_(manifold.proju(p, v))
