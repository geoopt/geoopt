import torch.optim
from .mixin import RiemannianOptimMixin
from ..manifolds import Rn


class RiemannianSGD(torch.optim.SGD, RiemannianOptimMixin):
    """Riemannian Stochastic Gradient Descent"""

    def __init__(self, *args, manifold=Rn(), proj_x_every=1000, **kwargs):
        super().__init__(*args, **kwargs)
        RiemannianOptimMixin.__init__(self, manifold=manifold, proj_x_every=proj_x_every)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        proju = self.manifold.proju
        projx = self.manifold.projx
        retr = self.manifold.retr
        transp = self.manifold.transp
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if state['step'] % self.proj_x_every == 0:
                    p.data.set_(projx(p.data))
                d_p = proju(p.data, d_p)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # the very first step, d_p is already projected
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        # buf is already transported
                        buf = param_state['momentum_buffer']
                        if state['step'] % self.proj_x_every == 0:
                            # refining numerical issues
                            buf.data.set_(projx(buf.data))
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                    p.data.set_(retr(p.data, d_p, -group['lr']))
                    buf.data.set_(transp(p.data, d_p, buf, -group['lr']))
                else:
                    p.data.set_(retr(p.data, d_p, -group['lr']))
                state['step'] += 1
        return loss
