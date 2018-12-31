import torch.optim

from .mixin import OptimMixin
from .tracing import create_traced_update
from ..tensor import ManifoldParameter, ManifoldTensor
from ..manifolds import Euclidean


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""Riemannian Adam with the same API as :class:`torch.optim.Adam`

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)


    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

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
                betas = self.group_param_tensor(group, "betas")
                weight_decay = self.group_param_tensor(group, "weight_decay")
                eps = self.group_param_tensor(group, "eps")
                learning_rate = self.group_param_tensor(group, "lr")
                amsgrad = group["amsgrad"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                    else:
                        manifold = Euclidean()

                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = torch.tensor(0)
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
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
                    if "traced_step" not in state:
                        if amsgrad:
                            state["traced_step"] = create_traced_update(
                                self.perform_step,
                                manifold,
                                p,
                                weight_decay.type_as(p),
                                betas.type_as(p),
                                eps.type_as(p),
                                state["step"],
                                state["exp_avg"],
                                state["exp_avg_sq"],
                                state["max_exp_avg_sq"],
                                amsgrad=True,
                            )
                        else:
                            state["traced_step"] = create_traced_update(
                                self.perform_step,
                                manifold,
                                p,
                                weight_decay.type_as(p),
                                betas.type_as(p),
                                eps.type_as(p),
                                state["step"],
                                state["exp_avg"],
                                state["exp_avg_sq"],
                                max_exp_avg_sq=None,
                                amsgrad=False,
                            )
                    if amsgrad:
                        state["traced_step"](
                            p,
                            p.grad,
                            learning_rate.type_as(p),
                            weight_decay.type_as(p),
                            betas.type_as(p),
                            eps.type_as(p),
                            state["step"],
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            state["max_exp_avg_sq"],
                        )
                    else:
                        state["traced_step"](
                            p,
                            p.grad,
                            learning_rate.type_as(p),
                            weight_decay.type_as(p),
                            betas.type_as(p),
                            eps.type_as(p),
                            state["step"],
                            state["exp_avg"],
                            state["exp_avg_sq"],
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
        betas,
        eps,
        step,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        amsgrad,
    ):
        grad.add_(weight_decay, point)
        grad = manifold.proju(point, grad)
        exp_avg.mul_(betas[0]).add_(1 - betas[0], grad)
        exp_avg_sq.mul_(betas[1]).add_(1 - betas[1], manifold.inner(point, grad))
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(eps)
        else:
            denom = exp_avg_sq.sqrt().add_(eps)
        denom = manifold.broadcast_scalar(denom)
        step.add_(1)
        bias_correction1 = 1 - betas[0] ** step.type_as(betas)
        bias_correction2 = 1 - betas[1] ** step.type_as(betas)
        step_size = lr * bias_correction2.sqrt_().div_(bias_correction1)

        # copy the state, we need it for retraction
        # get the direction for ascend
        direction = exp_avg / denom
        # transport the exponential averaging to the new point
        new_point, exp_avg_new = manifold.retr_transp(
            point, direction, -step_size, exp_avg
        )
        point.set_(new_point)
        exp_avg.set_(exp_avg_new)

    def stabilize_group(self, group):
        with torch.no_grad():
            for p in group["params"]:
                if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    continue
                state = self.state[p]
                manifold = p.manifold
                exp_avg = state["exp_avg"]
                p.set_(manifold.projx(p))
                exp_avg.set_(manifold.proju(p, exp_avg))

    def _sanitize_group(self, group):
        group = group.copy()
        if isinstance(group["lr"], torch.Tensor):
            group["lr"] = group["lr"].item()
        if isinstance(group["weight_decay"], torch.Tensor):
            group["weight_decay"] = group["weight_decay"].item()
        if isinstance(group["eps"], torch.Tensor):
            group["eps"] = group["eps"].item()
        if isinstance(group["betas"], torch.Tensor):
            group["betas"] = group["betas"][0].item(), group["betas"][1].item()
        return group
