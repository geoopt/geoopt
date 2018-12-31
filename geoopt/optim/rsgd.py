import torch.optim.optimizer
from ..manifolds import Euclidean
from ..tensor import ManifoldParameter, ManifoldTensor
from .mixin import OptimMixin
from .tracing import create_traced_update


__all__ = ["RiemannianSGD"]


class RiemannianSGD(OptimMixin, torch.optim.Optimizer):
    r"""Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        use_momentum=None,
        stabilize=None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_momentum=use_momentum or bool(momentum),
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments
        ---------
        closure : callable (optional)
            A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = self.group_param_tensor(group, "weight_decay")
                momentum = self.group_param_tensor(group, "momentum")
                dampening = self.group_param_tensor(group, "dampening")
                nesterov = group["nesterov"]
                learning_rate = self.group_param_tensor(group, "lr")
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        if momentum != 0:
                            state["momentum_buffer"] = p.grad.clone()
                    if "traced_step" not in state:
                        if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                            manifold = p.manifold
                        else:
                            manifold = Euclidean()

                        if group["use_momentum"]:
                            state["traced_step"] = create_traced_update(
                                self.perform_step,
                                manifold,
                                p,
                                weight_decay.type_as(p),
                                momentum.type_as(p),
                                state["momentum_buffer"],
                                dampening=dampening,
                                nesterov=nesterov,
                                use_momentum=group["use_momentum"],
                            )
                        else:
                            state["traced_step"] = create_traced_update(
                                self.perform_step,
                                manifold,
                                p,
                                weight_decay.type_as(p),
                                momentum=None,
                                momentum_buffer=None,
                                dampening=dampening,
                                nesterov=nesterov,
                                use_momentum=group["use_momentum"],
                            )
                    if group["use_momentum"]:
                        state["traced_step"](
                            p,
                            p.grad,
                            learning_rate.type_as(p),
                            weight_decay.type_as(p),
                            momentum.type_as(p),
                            state["momentum_buffer"],
                        )
                    else:
                        state["traced_step"](
                            p, p.grad, learning_rate.type_as(p), weight_decay.type_as(p)
                        )
                group["step"] += 1
                if self._stabilize is not None and group["step"] % self._stabilize == 0:
                    self.stabilize_group(group)
        return loss

    @staticmethod
    def perform_step(
        manifold,
        point,
        grad,
        lr,
        weight_decay,
        momentum,
        momentum_buffer,
        dampening,
        nesterov,
        use_momentum,
    ):
        grad.add_(weight_decay, point)
        grad = manifold.proju(point, grad)
        if use_momentum:
            momentum_buffer.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad = grad.add_(momentum, momentum_buffer)
            else:
                grad = momentum_buffer
            # we have all the things projected
            new_point, new_momentum_buffer = manifold.retr_transp(
                point, grad, -lr, momentum_buffer
            )
            momentum_buffer.set_(new_momentum_buffer)
            point.set_(new_point)
        else:
            new_point = manifold.retr(point, grad, -lr)
            point.set_(new_point)

    def stabilize_group(self, group):
        with torch.no_grad():
            for p in group["params"]:
                if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    continue
                manifold = p.manifold
                momentum = group["momentum"]
                p.set_(manifold.projx(p))
                if momentum > 0:
                    param_state = self.state[p]
                    if "momentum_buffer" in param_state:
                        buf = param_state["momentum_buffer"]
                        buf.set_(manifold.proju(p, buf))

    def _sanitize_group(self, group):
        group = group.copy()
        if isinstance(group["weight_decay"], torch.Tensor):
            group["weight_decay"] = group["weight_decay"].item()
        if isinstance(group["dampening"], torch.Tensor):
            group["dampening"] = group["dampening"].item()
        if isinstance(group["momentum"], torch.Tensor):
            group["momentum"] = group["momentum"].item()
        return group
