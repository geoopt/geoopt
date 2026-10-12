r"""
Riemannian and gyrovector operations of the Proper Velocity (PV) model of hyperbolic space.

Points of the PV model :math:`\mathcal{PV}^n_k` are arbitrary vectors :math:`x \in \mathbb{R}^n`.
They are the space-like coordinates of the hyperboloid :math:`\{(x_0, x): -x_0^2 + \|x\|^2 = -k\}`
with :math:`x_0 = \sqrt{k + \|x\|^2}`, so the model has constant curvature :math:`-1/k`, the same
convention as :class:`geoopt.Lorentz`. Unlike the Poincaré ball or the hyperboloid, the model has
no boundary and no constraint.

All formulas use the relativistic beta factor :math:`\beta_x = 1 / \sqrt{1 + \|x\|^2 / k}`.

References
----------
Ziheng Chen, Zihan Su, Bernhard Schölkopf, Nicu Sebe. Proper Velocity Neural Networks. ICLR 2026.
"""

from typing import List, Optional

import torch.jit

from ...utils import drop_dims
from ..lorentz.math import _default_reducedim, _reducedim_without_manifold

__all__ = [
    "beta",
    "inner",
    "norm",
    "egrad2rgrad",
    "gyroadd",
    "gyroinv",
    "gyroscalar",
    "gyration",
    "expmap0",
    "logmap0",
    "dist0",
    "expmap",
    "logmap",
    "dist",
    "parallel_transport0",
    "parallel_transport0back",
    "parallel_transport",
    "geodesic",
    "frechet_mean",
    "frechet_variance",
    "pv_to_lorentz",
    "lorentz_to_pv",
    "pv_to_poincare",
    "poincare_to_pv",
]


@torch.jit.script
def _sqnorm(x, dim: int = -1):
    return (x * x).sum(dim=dim, keepdim=True)


@torch.jit.script
def _dot(x, y, dim: int = -1):
    return (x * y).sum(dim=dim, keepdim=True)


def beta(x, *, k, dim=-1):
    r"""
    Relativistic beta factor of the PV model.

    .. math::

        \beta_x = \frac{1}{\sqrt{1 + \|x\|^2 / k}}

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        beta factor, with ``dim`` kept
    """
    return _beta(x, k, dim=dim)


@torch.jit.script
def _beta(x, k: torch.Tensor, dim: int = -1):
    return torch.rsqrt(1.0 + _sqnorm(x, dim=dim) / k)


@torch.jit.script
def _inv_beta_minus_one(x, k: torch.Tensor, dim: int = -1):
    # 1 / beta - 1 = sqrt(1 + t) - 1 = t / (sqrt(1 + t) + 1), stable for small t
    t = _sqnorm(x, dim=dim) / k
    return t / (torch.sqrt(1.0 + t) + 1.0)


def inner(x, u, v, *, k, keepdim=False, dim=-1):
    r"""
    Riemannian inner product of tangent vectors at :math:`x`.

    The metric is the pullback of the Minkowski metric through
    :math:`x \mapsto (\sqrt{k + \|x\|^2}, x)`:

    .. math::

        \langle u, v \rangle_x = \langle u, v \rangle - \frac{\langle x, u \rangle \langle x, v \rangle}{k + \|x\|^2}

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    u : tensor
        tangent vector at :math:`x`
    v : tensor
        tangent vector at :math:`x`
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(x, u, v, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _inner(x, u, v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = _dot(u, v, dim=dim) - _dot(x, u, dim=dim) * _dot(x, v, dim=dim) / (
        k + _sqnorm(x, dim=dim)
    )
    if not keepdim:
        res = res.squeeze(dim)
    return res


def norm(x, u, *, k, keepdim=False, dim=-1):
    r"""
    Riemannian norm of a tangent vector at :math:`x`.

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    u : tensor
        tangent vector at :math:`x`
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of :math:`u`
    """
    return _norm(x, u, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _norm(x, u, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    return torch.sqrt(_inner(x, u, u, k, keepdim=keepdim, dim=dim).clamp_min(1e-30))


def egrad2rgrad(x, grad, *, k, dim=-1):
    r"""
    Convert the Euclidean gradient to the Riemannian gradient (inverse metric).

    .. math::

        \operatorname{grad} f(x) = \nabla f(x) + \frac{\langle x, \nabla f(x) \rangle}{k} x

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    grad : tensor
        Euclidean gradient at :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        Riemannian gradient
    """
    return _egrad2rgrad(x, grad, k, dim=dim)


@torch.jit.script
def _egrad2rgrad(x, grad, k: torch.Tensor, dim: int = -1):
    return grad + _dot(x, grad, dim=dim) / k * x


def gyroadd(x, y, *, k, dim=-1):
    r"""
    PV gyroaddition.

    .. math::

        x \oplus y = x + y + \left(\frac{\beta_x}{1 + \beta_x} \frac{\langle x, y \rangle}{k}
        + \frac{1 - \beta_y}{\beta_y}\right) x

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    y : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        gyrosum of :math:`x` and :math:`y`
    """
    return _gyroadd(x, y, k, dim=dim)


@torch.jit.script
def _gyroadd(x, y, k: torch.Tensor, dim: int = -1):
    bx = _beta(x, k, dim=dim)
    coef = bx / (1.0 + bx) * _dot(x, y, dim=dim) / k + _inv_beta_minus_one(y, k, dim=dim)
    return x + y + coef * x


def gyroinv(x, *, dim=-1):
    r"""
    PV gyroinverse, :math:`\ominus x = -x`.

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    dim : int
        unused, kept for API consistency

    Returns
    -------
    tensor
        gyroinverse of :math:`x`
    """
    return -x


def gyroscalar(r, x, *, k, dim=-1):
    r"""
    PV gyro scalar multiplication.

    .. math::

        r \otimes x = \sqrt{k} \sinh\left(r \operatorname{arsinh}\frac{\|x\|}{\sqrt{k}}\right)
        \frac{x}{\|x\|} = \operatorname{Exp}_0(r \operatorname{Log}_0(x))

    Parameters
    ----------
    r : tensor
        scalar, broadcastable to ``x`` with ``dim`` kept
    x : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        :math:`r \otimes x`
    """
    return _expmap0(r * _logmap0(x, k, dim=dim), k, dim=dim)


def expmap0(u, *, k, dim=-1):
    r"""
    Exponential map at the origin.

    .. math::

        \operatorname{Exp}_0(u) = \sqrt{k} \sinh\left(\frac{\|u\|}{\sqrt{k}}\right) \frac{u}{\|u\|}

    Parameters
    ----------
    u : tensor
        tangent vector at the origin
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on the PV manifold
    """
    return _expmap0(u, k, dim=dim)


@torch.jit.script
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    sqrt_k = torch.sqrt(k)
    u_norm = torch.norm(u, p=2, dim=dim, keepdim=True)
    arg = u_norm / sqrt_k
    # sinh(t) / t -> 1 for t -> 0
    coef = torch.where(arg > 1e-7, torch.sinh(arg) / arg.clamp_min(1e-15), torch.ones_like(arg))
    return coef * u


def logmap0(y, *, k, dim=-1):
    r"""
    Logarithmic map at the origin.

    .. math::

        \operatorname{Log}_0(y) = \sqrt{k} \operatorname{arsinh}\left(\frac{\|y\|}{\sqrt{k}}\right)
        \frac{y}{\|y\|}

    Parameters
    ----------
    y : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        tangent vector at the origin
    """
    return _logmap0(y, k, dim=dim)


@torch.jit.script
def _logmap0(y, k: torch.Tensor, dim: int = -1):
    sqrt_k = torch.sqrt(k)
    y_norm = torch.norm(y, p=2, dim=dim, keepdim=True)
    arg = y_norm / sqrt_k
    # arsinh(t) / t -> 1 for t -> 0
    coef = torch.where(arg > 1e-7, torch.asinh(arg) / arg.clamp_min(1e-15), torch.ones_like(arg))
    return coef * y


def dist0(y, *, k, keepdim=False, dim=-1):
    r"""
    Geodesic distance to the origin.

    .. math::

        d(0, y) = \sqrt{k} \operatorname{arsinh}\left(\frac{\|y\|}{\sqrt{k}}\right)

    Parameters
    ----------
    y : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        distance
    """
    return _dist0(y, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist0(y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    sqrt_k = torch.sqrt(k)
    return sqrt_k * torch.asinh(torch.norm(y, p=2, dim=dim, keepdim=keepdim) / sqrt_k)


def parallel_transport0(y, v, *, k, dim=-1):
    r"""
    Parallel transport from the origin to :math:`y`.

    .. math::

        \operatorname{PT}_{0 \to y}(v) = v + \frac{\beta_y}{1 + \beta_y} \frac{\langle y, v \rangle}{k} y

    Parameters
    ----------
    y : tensor
        target point on the PV manifold
    v : tensor
        tangent vector at the origin
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        tangent vector at :math:`y`
    """
    return _parallel_transport0(y, v, k, dim=dim)


@torch.jit.script
def _parallel_transport0(y, v, k: torch.Tensor, dim: int = -1):
    by = _beta(y, k, dim=dim)
    return v + by / (1.0 + by) * _dot(y, v, dim=dim) / k * y


def parallel_transport0back(x, v, *, k, dim=-1):
    r"""
    Parallel transport from :math:`x` to the origin, the inverse of :func:`parallel_transport0`.

    .. math::

        \operatorname{PT}_{x \to 0}(v) = v - \frac{\beta_x^2}{1 + \beta_x} \frac{\langle x, v \rangle}{k} x

    Parameters
    ----------
    x : tensor
        start point on the PV manifold
    v : tensor
        tangent vector at :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        tangent vector at the origin
    """
    return _parallel_transport0back(x, v, k, dim=dim)


@torch.jit.script
def _parallel_transport0back(x, v, k: torch.Tensor, dim: int = -1):
    bx = _beta(x, k, dim=dim)
    return v - bx * bx / (1.0 + bx) * _dot(x, v, dim=dim) / k * x


def expmap(x, u, *, k, dim=-1):
    r"""
    Exponential map at :math:`x`.

    .. math::

        \operatorname{Exp}_x(u) = x \oplus \operatorname{Exp}_0(\operatorname{PT}_{x \to 0}(u))

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    u : tensor
        tangent vector at :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on the PV manifold
    """
    return _expmap(x, u, k, dim=dim)


@torch.jit.script
def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    return _gyroadd(x, _expmap0(_parallel_transport0back(x, u, k, dim=dim), k, dim=dim), k, dim=dim)


def logmap(x, y, *, k, dim=-1):
    r"""
    Logarithmic map at :math:`x`.

    .. math::

        \operatorname{Log}_x(y) = \operatorname{PT}_{0 \to x}(\operatorname{Log}_0(\ominus x \oplus y))

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    y : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        tangent vector at :math:`x`
    """
    return _logmap(x, y, k, dim=dim)


@torch.jit.script
def _logmap(x, y, k: torch.Tensor, dim: int = -1):
    return _parallel_transport0(x, _logmap0(_gyroadd(-x, y, k, dim=dim), k, dim=dim), k, dim=dim)


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Geodesic distance, :math:`d(x, y) = d(0, \ominus x \oplus y)`.

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    y : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        distance
    """
    return _dist(x, y, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    return _dist0(_gyroadd(-x, y, k, dim=dim), k, keepdim=keepdim, dim=dim)


def parallel_transport(x, y, v, *, k, dim=-1):
    r"""
    Parallel transport from :math:`x` to :math:`y` along the connecting geodesic.

    It equals :math:`\operatorname{PT}_{0 \to y} \circ \operatorname{gyr}[y, \ominus x] \circ
    \operatorname{PT}_{x \to 0}`, evaluated with the numerically stable closed form

    .. math::

        \operatorname{PT}_{x \to y}(v) = v + \frac{\langle y, v \rangle - x_0^{-1} y_0 \langle x, v \rangle}
        {k + x_0 y_0 - \langle x, y \rangle} (x + y),
        \qquad x_0 = \sqrt{k + \|x\|^2}, \; y_0 = \sqrt{k + \|y\|^2}

    Parameters
    ----------
    x : tensor
        start point on the PV manifold
    y : tensor
        target point on the PV manifold
    v : tensor
        tangent vector at :math:`x`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        tangent vector at :math:`y`
    """
    return _parallel_transport(x, y, v, k, dim=dim)


@torch.jit.script
def _parallel_transport(x, y, v, k: torch.Tensor, dim: int = -1):
    x0 = torch.sqrt(k + _sqnorm(x, dim=dim))
    y0 = torch.sqrt(k + _sqnorm(y, dim=dim))
    nom = _dot(y, v, dim=dim) - y0 / x0 * _dot(x, v, dim=dim)
    denom = k + x0 * y0 - _dot(x, y, dim=dim)
    return v + nom / denom * (x + y)


def gyration(a, b, x, *, k, dim=-1):
    r"""
    PV gyration :math:`\operatorname{gyr}[a, b] x = \ominus(a \oplus b) \oplus (a \oplus (b \oplus x))`.

    Gyrations of the PV model are linear isometries of :math:`\mathbb{R}^n`, so ``x`` may be either a
    point or a tangent vector. The map is evaluated as
    :math:`\operatorname{PT}_{a \to 0} \circ \operatorname{PT}_{\ominus b \to a} \circ \operatorname{PT}_{0 \to \ominus b}`,
    which equals the definition and stays accurate for large norms.

    Parameters
    ----------
    a : tensor
        point on the PV manifold
    b : tensor
        point on the PV manifold
    x : tensor
        point or vector
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        :math:`\operatorname{gyr}[a, b] x`
    """
    return _gyration(a, b, x, k, dim=dim)


@torch.jit.script
def _gyration(a, b, x, k: torch.Tensor, dim: int = -1):
    v = _parallel_transport0(-b, x, k, dim=dim)
    v = _parallel_transport(-b, a, v, k, dim=dim)
    return _parallel_transport0back(a, v, k, dim=dim)


def geodesic(t, x, y, *, k, dim=-1):
    r"""
    Point at time :math:`t` on the geodesic from :math:`x` (:math:`t=0`) to :math:`y` (:math:`t=1`).

    .. math::

        \gamma_{x \to y}(t) = x \oplus \left(t \otimes (\ominus x \oplus y)\right)

    Parameters
    ----------
    t : tensor
        time, broadcastable to ``x`` with ``dim`` kept
    x : tensor
        start point on the PV manifold
    y : tensor
        target point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on the geodesic
    """
    z = _gyroadd(-x, y, k, dim=dim)
    return _gyroadd(x, _expmap0(t * _logmap0(z, k, dim=dim), k, dim=dim), k, dim=dim)


def frechet_mean(
    x,
    *,
    k,
    reducedim: Optional[List[int]] = None,
    dim=-1,
    keepdim=False,
    max_iter=100,
    tol=1e-6,
):
    r"""
    Fréchet mean :math:`\arg\min_y \frac{1}{N} \sum_i d^2(x_i, y)`, computed with Karcher flow.

    The iteration starts from :math:`\operatorname{Exp}_0` of the mean of :math:`\operatorname{Log}_0(x_i)`
    and repeats

    .. math::

        \mu \leftarrow \operatorname{Exp}_\mu\left(\frac{1}{L} \frac{1}{N} \sum_i \operatorname{Log}_\mu(x_i)\right),
        \qquad L = \frac{1}{N} \sum_i \frac{d_i}{\sqrt{k}} \coth\frac{d_i}{\sqrt{k}}, \; d_i = d(\mu, x_i)

    where :math:`L \ge 1` bounds the Hessian of :math:`\frac{1}{2N} \sum_i d^2(x_i, \cdot)` at
    :math:`\mu`. The undamped Karcher flow (:math:`L = 1`) can oscillate for widely spread points;
    for concentrated points :math:`L \approx 1`. If ``tol`` is not reached within ``max_iter``
    iterations, the last iterate is returned.

    Parameters
    ----------
    x : tensor
        points on the PV manifold
    k : tensor
        manifold negative curvature
    reducedim : List[int]
        dimensions to average over, defaults to all dimensions except ``dim``
        (``dim`` is ignored if listed)
    dim : int
        manifold dimension
    keepdim : bool
        retain the reduced dimensions?
    max_iter : int
        maximum number of Karcher flow iterations
    tol : float
        stop once the Riemannian norm of the update is below ``tol``

    Returns
    -------
    tensor
        Fréchet mean
    """
    reducedim = _default_reducedim(x, reducedim, dim)
    dim = dim % x.dim()
    if len(reducedim) == 0:
        # the Fréchet mean of a single point is the point itself
        return x
    mean = _expmap0(_logmap0(x, k, dim=dim).mean(dim=reducedim, keepdim=True), k, dim=dim)
    sqrt_k = torch.sqrt(k)
    for _ in range(max_iter):
        grad = _logmap(mean, x, k, dim=dim).mean(dim=reducedim, keepdim=True)
        if bool((_norm(mean, grad, k, dim=dim) < tol).all()):
            break
        d = _dist(mean, x, k, keepdim=True, dim=dim) / sqrt_k
        # d * coth(d) -> 1 for d -> 0
        curv = torch.where(d > 1e-7, d / torch.tanh(d.clamp_min(1e-7)), torch.ones_like(d))
        mean = _expmap(mean, grad / curv.mean(dim=reducedim, keepdim=True), k, dim=dim)
    if not keepdim:
        mean = drop_dims(mean, reducedim)
    return mean


def frechet_variance(x, mean, *, k, reducedim: Optional[List[int]] = None, dim=-1, keepdim=False):
    r"""
    Fréchet variance :math:`\frac{1}{N} \sum_i d^2(x_i, \mu)`.

    Parameters
    ----------
    x : tensor
        points on the PV manifold
    mean : tensor
        Fréchet mean, either broadcastable to ``x`` or computed with ``keepdim=False``
    k : tensor
        manifold negative curvature
    reducedim : List[int]
        dimensions to average over, defaults to all dimensions except ``dim``
        (``dim`` is ignored if listed)
    dim : int
        manifold dimension
    keepdim : bool
        retain the reduced dimensions?

    Returns
    -------
    tensor
        Fréchet variance
    """
    reducedim = _default_reducedim(x, reducedim, dim)
    dim = dim % x.dim()
    if mean.dim() < x.dim():
        # mean was computed with keepdim=False
        for d in reducedim:
            mean = mean.unsqueeze(d)
    d2 = _dist(mean, x, k, keepdim=False, dim=dim) ** 2
    reducedim = _reducedim_without_manifold(reducedim, x.dim(), dim)
    if len(reducedim) > 0:
        d2 = d2.mean(dim=reducedim, keepdim=keepdim)
    return d2


def pv_to_lorentz(x, *, k, dim=-1):
    r"""
    Isometry from the PV model to the hyperboloid, :math:`x \mapsto (\sqrt{k + \|x\|^2}, x)`.

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on :class:`geoopt.Lorentz` with the same ``k``
    """
    return torch.cat((torch.sqrt(k + _sqnorm(x, dim=dim)), x), dim=dim)


def lorentz_to_pv(x, *, dim=-1):
    r"""
    Isometry from the hyperboloid to the PV model (drops the time coordinate).

    Parameters
    ----------
    x : tensor
        point on :class:`geoopt.Lorentz`
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on the PV manifold with the same ``k``
    """
    return x.narrow(dim, 1, x.size(dim) - 1)


def pv_to_poincare(x, *, k, dim=-1):
    r"""
    Isometry from the PV model to the Poincaré ball of curvature :math:`-1/k`.

    .. math::

        \pi(x) = \frac{\beta_x}{1 + \beta_x} x

    Parameters
    ----------
    x : tensor
        point on the PV manifold
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on ``geoopt.PoincareBall(c=1 / k)``
    """
    bx = _beta(x, k, dim=dim)
    return bx / (1.0 + bx) * x


def poincare_to_pv(x, *, k, dim=-1):
    r"""
    Isometry from the Poincaré ball of curvature :math:`-1/k` to the PV model, the inverse of
    :func:`pv_to_poincare`.

    .. math::

        \pi^{-1}(x) = \frac{2 x}{1 - \|x\|^2 / k}

    Parameters
    ----------
    x : tensor
        point on ``geoopt.PoincareBall(c=1 / k)``, i.e. :math:`\|x\| < \sqrt{k}`
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        point on the PV manifold
    """
    return 2.0 * x / (1.0 - _sqnorm(x, dim=dim) / k)
