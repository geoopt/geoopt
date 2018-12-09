import math
import torch.optim
from ..manifolds import Rn
from ..tensor import ManifoldParameter, ManifoldTensor
from .mixin import OptimMixin


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        stabilize = self._stabilize

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    manifold = p.manifold
                else:
                    manifold = Rn()
                proju = manifold.proju
                projx = manifold.projx
                retr = manifold.retr
                retr_transp = manifold.retr_transp
                inner = manifold.inner

                grad = p.grad.data
                # project gradient on tangent space of parameter
                grad = proju(p.data, grad)

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    inner_prod_shape = p.shape
                    if manifold.ndim > 0:
                        inner_prod_shape = inner_prod_shape[: -manifold.ndim]
                    state["exp_avg_sq"] = torch.zeros(
                        inner_prod_shape, dtype=p.dtype, device=p.device
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros(
                            inner_prod_shape, dtype=p.dtype, device=p.device
                        )

                # this is assumed to be already transported
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                # these vectors are on tangent space and can be combined
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, inner(p.data, grad))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                denom = manifold.broadcast_scalar(denom)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # copy the state, we need it for retraction
                # get the direction for ascend
                direction = exp_avg / denom
                # transport the exponential averaging to the new point
                new_p, exp_avg_new = retr_transp(p.data, direction, -step_size, exp_avg)
                p.data.set_(new_p)
                exp_avg.set_(exp_avg_new)
                # now all: point, exp-avg direction are already on manifold
                if stabilize is not None and (state["step"] - 1) % stabilize == 0:
                    p.data.set_(projx(p.data))
                    exp_avg.set_(proju(p.data, exp_avg))
        return loss

    def stabilize(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    continue
                state = self.state[p]
                manifold = p.manifold
                exp_avg = state["exp_avg"]
                p.data.set_(manifold.projx(p.data))
                exp_avg.set_(manifold.proju(p.data, exp_avg))
