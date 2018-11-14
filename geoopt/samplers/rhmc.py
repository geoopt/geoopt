import math
import numpy as np
import torch

from ..tensor import ManifoldParameter, ManifoldTensor
from ..manifolds import Rn
from ..optim.mixin import OptimMixin
from .hmc import HMC, SGHMC


class RHMC(OptimMixin, HMC):
    """Riemannian Hamiltonian Monte-Carlo"""

    def step(self, closure):
        """Performs a single sampling step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the log probability.
        """
        logp = closure()
        logp.backward()

        old_logp = logp.item()
        old_H = -old_logp
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    manifold = p.manifold
                else:
                    manifold = Rn()

                proju = manifold.proju
                retr_transp = manifold.retr_transp
                state = self.state[p]

                if 'r' not in state:
                    state['old_p'] = torch.zeros_like(p)
                    state['old_r'] = torch.zeros_like(p)
                    state['r'] = torch.zeros_like(p)
        
                r = state['r']
                r.normal_()
                r.set_(proju(p.data, r))

                old_H += 0.5 * (r * r).sum().item()
        
                state['old_p'].copy_(p.data)
                state['old_r'].copy_(r)
        
                epsilon = group['epsilon']

                r.add_(0.5 * epsilon * proju(p.data, p.grad))
                
                p_, r_ = retr_transp(p.data, r, epsilon)
                p.data.set_(p_)
                r.set_(r_)

                p.grad.data.zero_()

        for i in range(1, self.n_steps):
            logp = closure()
            logp.backward()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                    else:
                        manifold = Rn()

                    proju = manifold.proju
                    retr_transp = manifold.retr_transp

                    r = self.state[p]['r']
                    epsilon = group['epsilon']

                    r.add_(epsilon * proju(p.data, p.grad))
                    p_, r_ = retr_transp(p.data, r, epsilon)
                    p.data.set_(p_)
                    r.set_(r_)

                    p.grad.data.zero_()

        logp = closure()
        logp.backward()

        new_logp = logp.item()
        new_H = -new_logp

        #is_nan = False
        #is_inf = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    manifold = p.manifold
                else:
                    manifold = Rn()
                
                proju = manifold.proju

                r = self.state[p]['r']
                r.add_(0.5 * epsilon * proju(p.data, p.grad))
                p.grad.data.zero_()

                new_H += 0.5 * (r * r).sum().item()

                #is_nan = is_nan or np.isnan(p.cpu().detach().numpy()).any()
                #is_inf = is_inf or np.isnan(p.cpu().detach().numpy()).any()

        rho = min(1., math.exp(old_H - new_H))

        if not self.burnin:
            self.steps += 1
            self.acceptance_probs.append(rho)

        #if is_inf or is_nan or np.random.rand(1) >= rho: # reject
        if np.random.rand(1) >= rho: # reject
            if not self.burnin:
                self.n_rejected += 1
       
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    r = state['r']
                    p.data.copy_(state['old_p'])
                    r.copy_(state['old_r'])

            self.log_probs.append(old_logp)
        else:
            self.log_probs.append(new_logp)


class SGRHMC(SGHMC):
    """Stochastic Gradient Riemannian Hamiltonian Monte-Carlo"""

    def step(self, closure):
        """Performs a single sampling step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the log probability.
        """
        H_old = 0.
        H_new = 0.

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if 'v' not in state:
                    state['v'] = torch.zeros_like(q)

                epsilon = group['epsilon']
                v = state['v']
                v.normal_().mul_(epsilon)

                r = v / epsilon
                H_old += 0.5 * (r * r).sum().item()

        for i in range(self.n_steps + 1):
            logp = closure()
            logp.backward()

            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                            manifold = p.manifold
                        else:
                            manifold = Rn()
                        
                        proju = manifold.proju
                        retr_transp = manifold.retr_transp

                        epsilon, alpha = group['epsilon'], group['alpha']

                        v = self.state[p]['v']

                        p_, v_ = retr_transp(p.data, v, 1.)
                        p.data.set_(p_)
                        v.set_(v_)

                        n = proju(p.data, torch.randn_like(v))
                        v.mul_(1 - alpha).add_(epsilon * p.grad).add_(math.sqrt(2 * alpha * epsilon) * n)
                        p.grad.data.zero_()

                        r = v / epsilon
                        H_new += 0.5 * (r * r).sum().item()


        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())


    def stabilize(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    continue
                
                manifold = p.manifold
                v = self.state[p]['v']

                p.data.set_(manifold.projx(p.data))
                v.data.set_(manifold.proju(p.data, v))
