import torch.optim
from ..manifolds import Rn
from ..tensor import ManifoldParameter


class RiemannianSGD(torch.optim.SGD):
    """Riemannian Stochastic Gradient Descent"""

    def __init__(self, *args, stabilize=1000, **kwargs):
        # this should be called first to initialize defaults
        self.stabilize = stabilize
        super().__init__(*args, **kwargs)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            stabilize = self.stabilize

            for p in group["params"]:
                if isinstance(p, ManifoldParameter):
                    manifold = p.manifold
                else:
                    manifold = Rn()
                proju = manifold.proju
                projx = manifold.projx
                retr = manifold.retr
                transp = manifold.transp

                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if state["step"] % stabilize == 0:
                    p.data.set_(projx(p.data))
                d_p = proju(p.data, d_p)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        # the very first step, d_p is already projected
                        buf = param_state["momentum_buffer"] = d_p.clone()
                    else:
                        # buf is already transported
                        buf = param_state["momentum_buffer"]
                        if state["step"] % stabilize == 0:
                            # refining numerical issues
                            buf.data.set_(projx(buf.data))
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf.clone()
                    # we have all the things projected
                    buf.data.set_(transp(p.data, d_p, buf, -group["lr"]))
                    p.data.set_(retr(p.data, d_p, -group["lr"]))
                else:
                    p.data.set_(retr(p.data, d_p, -group["lr"]))
                state["step"] += 1
        return loss
