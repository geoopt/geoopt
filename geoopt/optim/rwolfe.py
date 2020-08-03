"""Riemannian Line Search.

This module implements line search on Riemannian manifolds using geoopt.
This module uses the same syntax as a Torch optimizer
"""

from .mixin import OptimMixin
from ..tensor import ManifoldParameter, ManifoldTensor
from scipy.optimize.linesearch import scalar_search_wolfe1, scalar_search_armijo
import warnings
import torch

__all__ = ["RiemannianLineSearch"]


class RiemannianLineSearch(OptimMixin, torch.optim.Optimizer):
    r"""Riemannian line search optimizer.

    If we try to minimize objective :math:`f\colon M\to \mathbb{R}`, then we take a
    step in the direction :math:`\eta=\mathrm{grad} f(x)`. We define objective function

    .. math::
        \phi(\alpha) = f(R_x(\alpha\eta)),

    where :math:`R_x` is the retraction at :math:`x`.
    The Wolfe conditions for the step size :math:`\alpha` are then given by

    .. math::
        f(R_x(\alpha\eta))\leq f(x)+c_1 \alpha \langle\mathrm{grad} f(x),\eta\rangle

    and

    .. math::
        \langle\mathrm{grad}f(R_x(\alpha\eta)),\mathcal T(\alpha \eta,\mathrm{grad}
        f(x))\rangle \geq c_2\langle\mathrm{grad}f(x),\eta\rangle,

    where :math:`\mathcal T` is the vector transport. In terms of the line search
    objective these are simply given by

    .. math::
        \phi(\alpha)\leq \phi(0)+c_1\alpha\phi'(0)

    and

    .. math::
        \phi'(\alpha)\geq c_2\phi'(0)

    The constants :math:`c_1` and :math:`c_2` satisfy :math:`c_1\in (0,1)`
    and :math:`c_2\in (c_1,1)`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    line_search_method : ('wolfe' or 'armijo')
        Flag whether to use (strong) Wolfe conditions for line search,
        or use Armijo backtracking. If `armijo` is chosen, parameters
        `c2` and `amax` are ignored. (default: 'wolfe')
    c1 : float
        Parameter controlling Armijo rule (default: 1e-4)
    c2 : float
        Parameter controlling curvature rule (default: 0.9)
    fallback_stepsize : float
        fallback_stepsize to take if no point can be found satisfying
        line search conditions. See also :meth:`step` (default: 1)
    amax : float
        maximum step size (default: 50)
    amin : float
        minimum step size (default: 1e-8)
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)

    Attributes
    ----------
    last_step_size : int or `None`
        Last step size taken. If `None` no suitable step size was
        found, and consequently no step was taken.
    step_size_history : List[int or `None`]
        List of all step sizes taken so far.
    """

    def __init__(
        self,
        params,
        line_search_method="wolfe",
        c1=1e-4,
        c2=0.9,
        fallback_stepsize=1,
        amax=50,
        amin=1e-8,
        stabilize=None,
    ):
        defaults = dict(
            line_search_method=line_search_method,
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
        self._params = []
        for group in self.param_groups:
            self._params.extend(group["params"])
        if len(self.param_groups) > 1:
            warning_string = """Multiple parameter groups detected.
            Line search parameters will be taken from first group.
            """
            warnings.warn(warning_string, UserWarning)
        self.line_search_method = self.param_groups[0]["line_search_method"]
        if self.line_search_method not in ("wolfe", "armijo"):
            raise ValueError(
                f"Unrecognized line search method '{self.line_search_method}'"
            )
        self.c1 = self.param_groups[0]["c1"]
        self.c2 = self.param_groups[0]["c2"]
        self.fallback_stepsize = self.param_groups[0]["fallback_stepsize"]
        self.amax = self.param_groups[0]["amax"]
        self.amin = self.param_groups[0]["amin"]
        self.old_phi = None
        self.step_size_history = []
        self.last_step_size = None
        self._last_step = None
        self._grads_computed = False
        self.prev_loss = None
        self.closure = None
        self._step_size_dic = dict()

    def _phi(self, step_size):
        """Compute the line search objective, and store its derivatives in the state."""

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

            state["der_phi"] = torch.sum(
                manifold.inner(point, -grad, state["grad_retr"])
            )
        self._grads_computed = True

        # roll back parameters to before step
        with torch.no_grad():
            for point, old_point in zip(self._params, param_copy):
                point.copy_(old_point)

        self._step_size_dic[step_size] = phi
        self._last_step = step_size
        return phi

    def _derphi(self, step_size):
        """Compute derivative of phi.

        The derivative of phi is given by computing inner
        product between all tensor gradients at target point and those at source point.
        The source gradients are transported to the target point, and both gradients are
        projected.
        """

        # Call _phi to compute gradients; Does nothing if _phi was
        # already called with this stepsize during this step
        self._phi(step_size)

        derphi = 0
        for point in self._params:
            state = self.state[point]
            if "der_phi" not in state:
                continue

            derphi += torch.sum(state["der_phi"])

        return derphi

    def _init_loss(self, recompute_gradients=False):
        """Compute loss and gradients at start of line search."""

        if recompute_gradients or (not self._grads_computed):
            loss = self.closure()
            reuse_grads = False
        else:
            loss = self.prev_loss
            reuse_grads = True

        derphi0 = 0
        self._step_size_dic = dict()

        for point in self._params:
            state = self.state[point]
            if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                manifold = point.manifold
            else:  # Use euclidean manifold
                manifold = self._default_manifold
            if reuse_grads:
                grad = state["new_grad"]
            else:
                grad = point.grad
                grad = manifold.egrad2rgrad(point, grad)

            grad_norm_squared = torch.sum(manifold.inner(point, grad)).item()
            state["grad"] = grad

            # contribution to phi'(0) is just grad norm squared
            derphi0 += -grad_norm_squared

        self._grads_computed = True

        return loss, derphi0

    def step(self, closure=None, force_step=False, recompute_gradients=False):
        """Do a line search step using strong wolfe or armijo conditions.

        Parameters
        ----------
        closure : callable
            A closure that reevaluates the model and returns the loss. 
            Optional for most optimizers.
        force_step : bool
            If `True`, take a unit step of size `self.fallback_stepsize`
            if no suitable step size can be found.
            If `False`, no step is taken in this situation. (default: `False`)
        recompute_gradients : bool
            If True, recompute the gradients. Use this if the parameters
            have changed in between consecutive steps. (default: False)
        """

        self.closure = closure
        phi0, derphi0 = self._init_loss(recompute_gradients=recompute_gradients)
        self._step_size_dic = dict()

        if self.line_search_method == "wolfe":
            step_size, new_phi, old_phi = scalar_search_wolfe1(
                self._phi,
                self._derphi,
                phi0=phi0,
                derphi0=derphi0,
                old_phi0=self.old_phi,
                c1=self.c1,
                c2=self.c2,
                amax=self.amax,
                amin=self.amin,
            )
        elif self.line_search_method == "armijo":
            if self.old_phi is None:
                alpha0 = self.fallback_stepsize
            else:  # Use previous function value to estimate initial step length
                alpha0 = 2 * (phi0 - self.old_phi) / derphi0
            step_size, new_phi = scalar_search_armijo(
                self._phi, phi0, derphi0, c1=self.c1, alpha0=alpha0, amin=self.amin,
            )
            old_phi = phi0

        self.step_size_history.append(step_size)
        self.last_step_size = step_size

        self.old_phi = old_phi

        # Ensure that the last step for which we computed the closure coincides with
        # proposed step size, so that we can reuse the gradients.
        if self._last_step != step_size:
            self._grads_computed = False

        # If it fails to find a good step, probably we have convergence
        if step_size is None:
            if force_step:
                step_size = self.fallback_stepsize
                self._grads_computed = False
            else:
                warning_string = """No suitable step size could be found, and no step
                was taken. Call `step` with `force_step=True` to take a step anyway.
                """
                warnings.warn(warning_string, UserWarning)

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

            # Use retract to perform the step
            new_point = manifold.retr(point, -1 * step_size * grad)

            with torch.no_grad():
                point.copy_(new_point)

                if (
                    self._stabilize is not None
                    and len(self.step_size_history) % self._stabilize == 0
                ):
                    point.copy_(manifold.projx(point))

        # Sometimes new_phi produced by scalar search is nonsense, use this instead.
        # We pull this value from a cache, so this is free
        if step_size is not None:
            new_loss = self._phi(step_size)
            self.prev_loss = new_loss
        else:
            new_loss = self.prev_loss
        return new_loss
