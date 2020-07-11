"""Riemannian Line Search

This module implements line search on Riemannian manifolds using geoopt.
This module uses the same syntax as a Torch optimizer"""

from .mixin import OptimMixin
from ..tensor import ManifoldParameter, ManifoldTensor
from scipy.optimize.linesearch import scalar_search_wolfe1
import torch

__all__ = ["RiemannianLineSearch"]

class RiemannianLineSearch(OptimMixin, torch.optim.Optimizer):
    r"""Riemannian line search optimizer using strong Wolfe conditions.
    If we try to minimize objective $f:M\to \mathbb{R}$, then we take a step in
    the direction $\eta=\mathrm{grad} f(x)$. We define objective function
    $$\phi(\alpha) = f(R_x(\alpha\eta))$$, where $R_x$ is the retraction at $x$.
    The Wolfe conditions for the step size $\alpha$ are then given by
    $$f(R_x(\alpha\eta))\leq f(x)+c_1 \alpha \langle\mathrm{grad} f(x),\eta\rangle$$
    and
    $$\langle\mathrm{grad}f(R_x(\alpha\eta)),\mathcal T(\alpha \eta,\grad f(x))\rangle
    \geq c_2\langle\mathrm{grad}f(x),\eta\rangle,$$
    where $\mathcal T$ is the vector transport. In terms of the line search objective
    these are simply given by
    $$\phi(\alpha)\leq \phi(0)+c_1\alpha\phi'(0)$$
    and
    $$\phi'(\alpha)\geq c_2\phi'(0)$$

    The constants $c_1$ and $c_2$ satisfy $c_1\in (0,1)$ and $c_2\in (c_1,1)$.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    c1 : float
        Parameter controlling Armijo rule (default: 1e-4)
    c2 : float
        Parameter controlling curvature rule (default: 0.9)
    fallback_stepsize : float
        fallback_stepsize to take if no point can be found satisfying the
        Wolfe conditions (default: 1)
    amax : float
        maximum step size (default: 50)
    amin : float
        minimum step size (default: 1e-8)
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        stabilize=None,
        c1=1e-4,
        c2=0.9,
        fallback_stepsize=1,
        amax=50,
        amin=1e-8
    ):
        defaults = dict(
            c1=c1,
            c2=c2,
            fallback_stepsize=fallback_stepsize,
            amax=amax,
            amin=amin,
            stabilize=stabilize,
        )
        super(RiemannianLineSearch, self).__init__(
            params, defaults, stabilize=stabilize
        )
        self._params = self.param_groups[0]["params"]
        self.c1 = self.param_groups[0]["c1"]
        self.c2 = self.param_groups[0]["c2"]
        self.fallback_stepsize = self.param_groups[0]["fallback_stepsize"]
        self.amax = self.param_groups[0]["amax"]
        self.amin = self.param_groups[0]["amin"]
        self.old_phi = None
        self.step_size_history = []
        self.closure = None
        self._step_size_dic = dict()

    def phi_(self, step_size):
        "compute the line search objective, and store its derivatives in the state"

        if step_size in self._step_size_dic:
            return self._step_size_dic[step_size]

        param_copy = [param.clone() for param in self._params]

        for point in self._params:
            state = self.state[point]
            if "grad" not in state:
                continue
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = self._default_manifold

            grad = state["grad"]

            # compute retract and transport in gradient direction
            new_point, grad_retr = manifold.retr_transp(point, -step_size * grad, grad)
            with torch.no_grad():
                point.copy_(new_point)

            state["grad_retr"] = grad_retr

        # recompute loss at new point
        phi = self.closure()

        # Store new gradients in state
        for point in self._params:
            grad = point.grad
            if grad is None:
                continue
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = self._default_manifold

            state = self.state[point]

            # project gradient onto tangent space
            grad = manifold.egrad2rgrad(point, grad)

            state["new_grad"] = grad

        # roll back parameters to before step
        with torch.no_grad():
            for point, old_point in zip(self._params, param_copy):
                point.copy_(old_point)

        self._step_size_dic[step_size] = phi
        return phi

    def derphi_(self, step_size):
        """Compute derivative of phi. The derivative of phi is given by computing inner
        product between all tensor gradients at target point and those at source point.
        The source gradients are transported to the target point, and both gradients are
        projected."""

        # Call phi_ to compute gradients; Does nothing if phi_ was
        # already called with this stepsize during this step
        self.phi_(step_size)

        derphi = 0
        for point in self._params:
            state = self.state[point]
            if "grad" not in state:
                continue

            # For some reason using the metric for inner product gives wrong results,
            # therefore us euclidean product instead.
            derphi += torch.tensordot(
                -state["new_grad"],
                state["grad_retr"],
                dims=len(state["new_grad"].shape),
            ).item()
        return derphi

    def init_loss(self, closure):
        "Compute loss and gradients at start of line search"

        loss = closure()
        derphi0 = 0
        grad_norms = []
        self.step_size_history = []
        self._step_size_dic = dict()

        for point in self._params:
            grad = point.grad
            if grad is None:
                continue
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = self._default_manifold

            state = self.state[point]
            # project gradient onto tangent space
            grad = manifold.egrad2rgrad(point, grad)
            grad_norms.append(grad.norm().item())
            state["grad"] = grad
            state["loss"] = loss

            # contribution to phi'(0) is just grad norm squared
            derphi0 += -((grad.norm().item()) ** 2)

        return loss, derphi0

    def step(self, closure):
        """Do a line search step using strong wolfe conditions. If no suitable stepsize
        can be computed, do unit step."""
        self.closure = closure
        phi0, derphi0 = self.init_loss(closure)
        self._step_size_dic = dict()

        step_size, new_phi, old_phi = scalar_search_wolfe1(
            self.phi_,
            self.derphi_,
            phi0=phi0,
            derphi0=derphi0,
            old_phi0=self.old_phi,
            c1=self.c1,
            c2=self.c2,
            amax=self.amax,
            amin=self.amin,
        )
        self.step_size_history.append(step_size)

        self.old_phi = old_phi

        # If it fails to find a good step, just do a unit sized step
        if step_size is None:
            step_size = self.fallback_stepsize

        for point in self._params:
            if step_size is None:
                continue

            state = self.state[point]
            if "grad" not in state:
                continue
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = self._default_manifold

            grad = state["grad"]

            # compute retract and transport in gradient direction
            new_point = manifold.retr(point, -1 * step_size * grad)

            with torch.no_grad():
                point.copy_(new_point)

        # Cast scalar to a scalar Torch tensor
        new_closure = torch.Tensor(1) + new_phi
        return new_closure
