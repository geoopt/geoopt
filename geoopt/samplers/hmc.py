import math
import numpy as np
import torch
from .base import Sampler


class HMC(Sampler):
    """Hamiltonian Monte-Carlo"""

    def __init__(self, params, epsilon=1e-3, n_steps=1):
        defaults = dict(epsilon=epsilon)
        super(HMC, self).__init__(params, defaults)
        self.n_steps = n_steps
    
    
    def step(self, closure, *args):
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

                state = self.state[p]

                if 'r' not in state:
                    state['old_p'] = torch.empty_like(p)
                    state['old_r'] = torch.empty_like(p)
                    state['r'] = torch.empty_like(p)
        
                r = state['r']
                r.normal_()

                old_H += 0.5 * (r * r).sum().item()
        
                state['old_p'].copy_(p)
                state['old_r'].copy_(r)
        
                epsilon = group['epsilon']

                r.add_(0.5 * epsilon * p.grad)
                p.data.add_(epsilon * r)
                p.grad.data.zero_()

        for i in range(1, self.n_steps):
            logp = closure()
            logp.backward()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    r = self.state[p]['r']
                    epsilon = group['epsilon']

                    r.add_(epsilon * p.grad)
                    p.data.add_(epsilon * r)
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

                r = self.state[p]['r']
                r.add_(0.5 * epsilon * p.grad)
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


class SGHMC(Sampler):
    """Stochastic Gradient Hamiltonian Monte-Carlo"""

    def __init__(self, params, epsilon=1e-3, n_steps=1, alpha=0.1):
        defaults = dict(epsilon=epsilon, alpha=alpha)
        super(SGHMC, self).__init__(params, defaults)
        self.n_steps = n_steps
    
    
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
                    state['v'] = torch.zeros_like(p)

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
                        state = self.state[p]
                        v = state['v']

                        epsilon, alpha = group['epsilon'], group['alpha']
                        p.data.add_(v)

                        n = torch.randn_like(v)
                        v.mul_(1 - alpha).add_(epsilon * p.grad).add_(math.sqrt(2 * alpha * epsilon) * n)
                        p.grad.data.zero_()

                        r = v / epsilon
                        H_new += 0.5 * (r * r).sum().item()


        if not self.burnin:
            self.steps += 1
            self.log_probs.append(logp.item())
