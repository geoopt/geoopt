import math

import torch

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.samplers.base import Sampler

__all__ = ["SGRHMC"]


class SGRHMC(Sampler):
    r"""
    Stochastic Gradient Riemannian Hamiltonian Monte-Carlo.

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
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.n_steps = n_steps

    def step(self, closure):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                if "v" not in state:
                    state["v"] = torch.zeros_like(p)

                epsilon = group["epsilon"]
                v = state["v"]
                v.normal_().mul_(epsilon)

                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    with torch.no_grad():
                        v.copy_(p.manifold.proju(p, v))
        for _ in range(self.n_steps):
            with torch.enable_grad():
                logp = closure()
                logp.backward()
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                            manifold = p.manifold
                        else:
                            manifold = self._default_manifold

                        egrad2rgrad = manifold.egrad2rgrad
                        retr_transp = manifold.retr_transp

                        epsilon, alpha = group["epsilon"], group["alpha"]

                        v = self.state[p]["v"]

                        n = egrad2rgrad(p, torch.randn_like(v))
                        v.mul_(1 - alpha).add_(egrad2rgrad(p, p.grad), alpha=epsilon).add_(
                            math.sqrt(2 * alpha * epsilon) * n
                        )
                        p.grad.zero_()

                        p_, v_ = retr_transp(p, v, v)
                        p.copy_(p_)
                        v.copy_(v_)

        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue

            manifold = p.manifold
            p.copy_(manifold.projx(p))
            # proj here is ok
            state = self.state[p]
            if not state:
                continue
            state["v"].copy_(manifold.proju(p, state["v"]))
