import math

import numpy as np
import torch

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Euclidean
from geoopt.samplers.base import Sampler


__all__ = ["RHMC"]


class RHMC(Sampler):
    """Riemannian Hamiltonian Monte-Carlo

    Parameters
    ----------
    params : iterable[tensor]
        iterables of tensors for which to perform sampling
    epsilon : float
        step size
    n_steps : int
        number of leapfrog steps
    """

    def __init__(self, params, epsilon=1e-3, n_steps=1):
        defaults = dict(epsilon=epsilon)
        super().__init__(params, defaults)
        self.n_steps = n_steps

    def _step(self, p, r, epsilon):
        if isinstance(p, (ManifoldParameter, ManifoldTensor)):
            manifold = p.manifold
        else:
            manifold = Euclidean()

        proju = manifold.proju
        retr_transp = manifold.retr_transp

        r.add_(epsilon * proju(p.data, p.grad))
        p_, r_ = retr_transp(p.data, r, epsilon, r)
        p.data.set_(p_)
        r.set_(r_)

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

        old_logp = logp.item()
        old_H = -old_logp

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    manifold = p.manifold
                else:
                    manifold = Euclidean()

                proju = manifold.proju
                state = self.state[p]

                if "r" not in state:
                    state["old_p"] = torch.zeros_like(p)
                    state["old_r"] = torch.zeros_like(p)
                    state["r"] = torch.zeros_like(p)

                r = state["r"]
                r.normal_()
                r.set_(proju(p.data, r))

                old_H += 0.5 * (r * r).sum().item()

                state["old_p"].copy_(p.data)
                state["old_r"].copy_(r)

                epsilon = group["epsilon"]
                self._step(p, r, epsilon)
                p.grad.data.zero_()

        for i in range(1, self.n_steps):
            logp = closure()
            logp.backward()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    self._step(p, self.state[p]["r"], group["epsilon"])
                    p.grad.data.zero_()

        logp = closure()
        logp.backward()

        new_logp = logp.item()
        new_H = -new_logp

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    manifold = p.manifold
                else:
                    manifold = Euclidean()

                proju = manifold.proju

                r = self.state[p]["r"]
                r.add_(0.5 * epsilon * proju(p.data, p.grad))
                p.grad.data.zero_()

                new_H += 0.5 * (r * r).sum().item()

        rho = min(1.0, math.exp(old_H - new_H))

        if not self.burnin:
            self.steps += 1
            self.acceptance_probs.append(rho)

        if np.random.rand(1) >= rho:  # reject
            if not self.burnin:
                self.n_rejected += 1

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    r = state["r"]
                    p.data.copy_(state["old_p"])
                    r.copy_(state["old_r"])

            self.log_probs.append(old_logp)
        else:
            self.log_probs.append(new_logp)
