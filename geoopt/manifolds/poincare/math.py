"""
Poincare manifold utility functions.

Functions for math on Poincare ball model. Most of this is taken from
a well written paper by Octavian-Eugen Ganea (2018) [1]_.


.. [1] Octavian-Eugen Ganea et al., Hyperbolic Neural Networks, NIPS 2018
"""

import functools
import torch.jit


MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


def tanh(x):
    return x.clamp(-15, 15).tanh()


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(MIN_NORM).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def artanh(x):
    return Artanh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    k_zero = k.isclose(k.new_zeros(()))
    # shrink sign
    k_sign[k_zero] = 0
    k_sqrt = k.abs().sqrt()
    scaled_x = x * k_sqrt
    return (
        torch.where(k_zero, k.new_zeros(()), k_sqrt.reciprocal())
        * (
            torch.where(k_sign.gt(0), scaled_x.tan(), k.new_zeros(()))
            + torch.where(k_sign.lt(0), tanh(scaled_x), k.new_zeros(()))
        )
        + x * k_zero
    )


def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    k_zero = k.isclose(k.new_zeros(()))
    # shrink sign
    k_sign[k_zero] = 0
    k_sqrt = k.abs().sqrt()
    scaled_x = x * k_sqrt
    return (
        torch.where(k_zero, k.new_zeros(()), k_sqrt.reciprocal())
        * (
            torch.where(k_sign.gt(0), scaled_x.atan(), k.new_zeros(()))
            + torch.where(k_sign.lt(0), artanh(scaled_x), k.new_zeros(()))
        )
        + x * k_zero
    )


def arsin_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    k_zero = k.isclose(k.new_zeros(()))
    # shrink sign
    k_sign[k_zero] = 0
    k_sqrt = k.abs().sqrt()
    scaled_x = x * k_sqrt
    return (
        torch.where(k_zero, k.new_zeros(()), k_sqrt.reciprocal())
        * (
            torch.where(k_sign.gt(0), scaled_x.asin(), k.new_zeros(()))
            + torch.where(k_sign.lt(0), arsinh(scaled_x), k.new_zeros(()))
        )
        + x * k_zero
    )


def project(x, *, k=0, dim=-1, eps=None):
    r"""
    Safe projection on the manifold for numerical stability.

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension to compute norm
    eps : float
        stability parameter, uses default for dtype if not provided

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k, dim, eps)


def _project(x, k, dim: int = -1, eps: float = None):
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    if eps is None:
        eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (k.abs() ** 0.5)
    maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), float("inf")))
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x, *, k=0, keepdim=False, dim=-1):
    r"""
    Compute the conformal factor :math:`\lambda^c_x` for a point on the ball.

    .. math::
        \lambda^\kappa_x = \frac{1}{1 + \kappa \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : float|tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        conformal factor
    """
    return _lambda_x(x, k, keepdim=keepdim, dim=dim)


def _lambda_x(x, k, keepdim: bool = False, dim: int = -1):
    return 2 / (1 + k * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(MIN_NORM)


def inner(x, u, v, *, k=0, keepdim=False, dim=-1):
    r"""
    Compute inner product for two vectors on the tangent space w.r.t Riemannian metric on the Poincare ball.

    .. math::

        \langle u, v\rangle_x = (\lambda^\kappa_x)^2 \langle u, v \rangle

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on Poincare ball
    v : tensor
        tangent vector to :math:`x` on Poincare ball
    k : float|tensor
        sectional curvature of manifold
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


def _inner(x, u, v, k, keepdim: bool = False, dim: int = -1):
    return _lambda_x(x, k, keepdim=True, dim=dim) ** 2 * (u * v).sum(
        dim=dim, keepdim=keepdim
    )


def norm(x, u, *, k=0, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Poincare ball.

    .. math::

        \|u\|_x = \lambda^\kappa_x \|u\|_2

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on Poincare ball
    k : float|tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(x, u, k, keepdim=keepdim, dim=dim)


def _norm(x, u, k, keepdim: bool = False, dim: int = -1):
    return _lambda_x(x, k, keepdim=keepdim, dim=dim) * u.norm(
        dim=dim, keepdim=keepdim, p=2
    )


def mobius_add(x, y, *, k=0, dim=-1):
    r"""
    Computes the Möbius gyrovector addition.

    .. math::

        x \oplus_\kappa y =
        \frac{
            (1 - 2 \kappa \langle x, y\rangle - \kappa \|y\|^2_2) x +
            (1 + \kappa \|x\|_2^2) y
        }{
            1 - 2 \kappa \langle x, y\rangle + \kappa^2 \|x\|^2_2 \|y\|^2_2
        }

    .. plot:: plots/extended/universal/mobius_add.py

    In general this operation is not commutative:

    .. math::

        x \oplus_\kappa y \ne y \oplus_\kappa x

    But in some cases this property holds:

    * zero vector case

    .. math::

        \mathbf{0} \oplus_\kappa x = x \oplus_\kappa \mathbf{0}

    * zero curvature case that is same as Euclidean addition

    .. math::

        x \oplus_0 y = y \oplus_0 x

    Another useful property is so called left-cancellation law:

    .. math::

        (-x) \oplus_\kappa (x \oplus_\kappa y) = y

    Parameters
    ----------
    x : tensor
        point on the manifold
    y : tensor
        point on the manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius addition
    """
    return _mobius_add(x, y, k, dim=dim)


def _mobius_add(x, y, k, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k ** 2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
    # 1)
    # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
    # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
    # 2)
    # {y + x * <y, y> = 0
    # {x + y * <x, x> = 0
    # 3)
    # {- y/<y, y> = x
    # {- x/<x, x> = y
    # 4)
    # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
    return num / denom.clamp_min(MIN_NORM)


def mobius_sub(x, y, *, k=0, dim=-1):
    r"""
    Computes the Möbius gyrovector subtraction.

    The Möbius subtraction can be represented via the Möbius addition as
    follows:

    .. math::

        x \ominus_\kappa y = x \oplus_\kappa (-y)

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius subtraction
    """
    return _mobius_sub(x, y, k, dim=dim)


def _mobius_sub(x, y, k, dim: int = -1):
    return _mobius_add(x, -y, k, dim=dim)


def mobius_coadd(x, y, *, k=0, dim=-1):
    r"""
    Computes the Möbius gyrovector coaddition.

    The addition operation :math:`\oplus_\kappa` is neither associative, nor
    commutative. In contrast, the coaddition :math:`\boxplus_\kappa` (or
    cooperation) is an associative operation that is defined as follows.

    .. math::

        a \boxplus_\kappa b
        =
        b \boxplus_\kappa a
        =
        a\operatorname{gyr}[a, -b]b\\
        = \frac{
            (1 + \kappa \|y\|^2_2) x + (1 + \kappa \|x\|_2^2) y
            }{
            1 + \kappa^2 \|x\|^2_2 \|y\|^2_2
        },

    where :math:`\operatorname{gyr}[a, b]v = \ominus_\kappa (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa v))`

    The following right cancellation property holds

    .. math::

        (a \boxplus_\kappa b) \ominus_\kappa b = a\\
        (a \oplus_\kappa b) \boxminus_\kappa b = a

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius coaddition

    """
    return _mobius_coadd(x, y, k, dim=dim)


# TODO: check numerical stability with Gregor's paper!!!
def _mobius_coadd(x, y, k, dim: int = -1):
    # x2 = x.pow(2).sum(dim=dim, keepdim=True)
    # y2 = y.pow(2).sum(dim=dim, keepdim=True)
    # num = (1 + K * y2) * x + (1 + K * x2) * y
    # denom = 1 - K ** 2 * x2 * y2
    # avoid division by zero in this way
    # return num / denom.clamp_min(MIN_NORM)
    #
    return _mobius_add(x, _gyration(x, -y, y, k=k, dim=dim), k, dim=dim)


def mobius_cosub(x, y, *, k=0, dim=-1):
    r"""
    Computes the Möbius gyrovector cosubtraction.

    The Möbius cosubtraction is defined as follows:

    .. math::

        a \boxminus_\kappa b = a \boxplus_\kappa -b

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius cosubtraction

    """
    return _mobius_cosub(x, y, k, dim=dim)


def _mobius_cosub(x, y, K, dim: int = -1):
    return _mobius_coadd(x, -y, K, dim=dim)


# TODO: can we make this operation somehow safer by breaking up the
# TODO: scalar multiplication for K>0 when the argument to the
# TODO: tan function gets close to pi/2+k*pi for k in Z?
# TODO: one could use the scalar associative law
# TODO: s_1 (X) s_2 (X) x = (s_1*s_2) (X) x
# TODO: to implement a more stable Möbius scalar mult
def mobius_scalar_mul(r, x, *, k=0, dim=-1):
    r"""
    Computes the Möbius scalar multiplication.

    .. math::

        r \otimes_\kappa x
        =
        \tan_\kappa(r\tan_\kappa^{-1}(\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_\kappa x = x \oplus_\kappa \dots \oplus_\kappa x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_\kappa x
         =
         r_1 \otimes_\kappa x \oplus r_2 \otimes_\kappa x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_\kappa x = r_1 \otimes_\kappa (r_2 \otimes_\kappa x)

    * Monodistributivity

    .. math::

         r \otimes_\kappa (r_1 \otimes x \oplus r_2 \otimes x) =
         r \otimes_\kappa (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_\kappa x / \|r \otimes_\kappa x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : float|tensor
        scalar for multiplication
    x : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    """
    return _mobius_scalar_mul(r, x, k, dim=dim)


def _mobius_scalar_mul(r, x, k, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_k(r * artan_k(x_norm, k), k) * (x / x_norm)
    return res_c


def dist(x, y, *, k=0, keepdim=False, dim=-1):
    r"""
    Computes the geodesic distance between :math:`x` and :math:`y` on the
    manifold.

    .. math::

        d_\kappa(x, y) = 2\tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)

    .. plot:: plots/extended/universal/distance.py

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k, keepdim=keepdim, dim=dim)


def _dist(x, y, k, keepdim: bool = False, dim: int = -1):
    return 2.0 * artan_k(
        _mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
    )


def dist0(x, *, k=0, keepdim=False, dim=-1):
    r"""
    Computes geodesic distance to the manifold's origin.

    Parameters
    ----------
    x : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    """
    return _dist0(x, k, keepdim=keepdim, dim=dim)


def _dist0(x, k, keepdim: bool = False, dim: int = -1):
    return 2.0 * artan_k(x.norm(dim=dim, p=2, keepdim=keepdim), k)


def geodesic(t, x, y, *, k=0, dim=-1):
    r"""
    Computes the point on the geodesic (shortest) path connecting :math:`x` and
    :math:`y` at time :math:`x`.

    The path can also be treated as an extension of the line segment to an
    unbounded geodesic that goes through :math:`x` and :math:`y`. The equation
    of the geodesic is given as:

    .. math::

        \gamma_{x\to y}(t)
        =
        x \oplus_\kappa t \otimes_\kappa ((-x) \oplus_\kappa y)

    The properties of the geodesic are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Furthermore, the geodesic also satisfies the property of local distance
    minimization:

    .. math::

         d_\kappa(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which
    yields the above formula with :math:`v=1`.

    However, we can always compute the constant speed :math:`v` from the points
    that the particular path connects:

    .. math::

        v = d_\kappa(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_\kappa(x, y)


    Parameters
    ----------
    t : float|tensor
        travelling time
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        point on the geodesic going through x and y
    """
    return _geodesic(t, x, y, k, dim=dim)


def _geodesic(t, x, y, k, dim: int = -1):
    # this is not very numerically stable
    v = _mobius_add(-x, y, k, dim=dim)
    tv = _mobius_scalar_mul(t, v, k, dim=dim)
    gamma_t = _mobius_add(x, tv, k, dim=dim)
    return gamma_t


def expmap(x, u, *, k=0, dim=-1):
    r"""
    Computes the exponential map of :math:`u` at :math:`x`.

    The expmap is tightly related with :func:`geodesic`. Intuitively, the
    expmap represents a smooth travel along a geodesic from the starting point
    :math:`x`, into the initial direction :math:`u` at speed :math:`\|u\|_x` for
    the duration of one time unit. In formulas one can express this as the
    travel along the curve :math:`\gamma_{x, u}(t)` such that

    .. math::

        \gamma_{x, u}(0) = x\\
        \dot\gamma_{x, u}(0) = u\\
        \|\dot\gamma_{x, u}(t)\|_{\gamma_{x, u}(t)} = \|u\|_x

    The existence of this curve relies on uniqueness of the differential
    equation solution, that is local. For the universal manifold the solution
    is well defined globally and we have.

    .. math::

        \operatorname{exp}^\kappa_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_\kappa \tan_\kappa(\|u\|_x/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    x : tensor
        starting point on manifold
    u : tensor
        speed vector in tangent space at x
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k, dim=dim)


def _expmap(x, u, k, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(x, k, dim=dim, keepdim=True)
    second_term = tan_k((lam / 2.0) * u_norm, k) * (u / u_norm)
    y = _mobius_add(x, second_term, k, dim=dim)
    return y


def expmap0(u, *, k=0, dim=-1):
    r"""
    Computes the exponential map of :math:`u` at the origin :math:`0`.

    .. math::

        \operatorname{exp}^\kappa_0(u)
        =
        \tan_\kappa(\|u\|_2/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    u : tensor
        speed vector on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


def _expmap0(u, k, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tan_k(u_norm, k) * (u / u_norm)
    return gamma_1


def geodesic_unit(t, x, u, *, k=0, dim=-1):
    r"""
    Computes the point on the unit speed geodesic at time :math:`t`, starting
    from :math:`x` with initial direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{x,u}(t) = x\oplus_\kappa \tan_\kappa(t/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point on manifold
    u : tensor
        initial direction in tangent space at x
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the point on the unit speed geodesic
    """
    return _geodesic_unit(t, x, u, k, dim=dim)


def _geodesic_unit(t, x, u, k, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = tan_k(t / 2.0, k) * (u / u_norm)
    gamma_1 = _mobius_add(x, second_term, k, dim=dim)
    return gamma_1


def logmap(x, y, *, k=0, dim=-1):
    r"""
    Computes the logarithmic map of :math:`y` at :math:`x`.

    .. math::

        \operatorname{log}^\kappa_x(y) = \frac{2}{\lambda_x^\kappa}
        \tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)
        * \frac{(-x)\oplus_\kappa y}{\|(-x)\oplus_\kappa y\|_2}

    The result of the logmap is a vector :math:`u` in the tangent space of
    :math:`x` such that

    .. math::

        y = \operatorname{exp}^\kappa_x(\operatorname{log}^\kappa_x(y))


    Parameters
    ----------
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_x M` that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k, dim=dim)


def _logmap(x, y, k, dim: int = -1):
    sub = _mobius_add(-x, y, k, dim=dim)
    sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(x, k, keepdim=True, dim=dim)
    return 2.0 * artan_k(sub_norm, k) * (sub / (lam * sub_norm))


def logmap0(y, *, k=0, dim=-1):
    r"""
    Computes the logarithmic map of :math:`y` at the origin :math:`0`.

    .. math::

        \operatorname{log}^\kappa_0(y)
        =
        \tan_\kappa^{-1}(\|y\|_2) \frac{y}{\|y\|_2}

    The result of the logmap at the origin is a vector :math:`u` in the tangent
    space of the origin :math:`0` such that

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Parameters
    ----------
    y : tensor
        target point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_0 M` that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k, dim=dim)


def _logmap0(y, K, dim: int = -1):
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    return (y / y_norm) * artan_k(y_norm, K)


def mobius_matvec(m, x, *, k=0, dim=-1):
    r"""
    Computes the generalization of matrix-vector multiplication in gyrovector
    spaces.

    The Möbius matrix vector operation is defined as follows:

    .. math::

        M \otimes_\kappa x = \tan_\kappa\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tan_\kappa^{-1}(\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}

    .. plot:: plots/extended/universal/mobius_matvec.py

    Parameters
    ----------
    m : tensor
        matrix for multiplication. Batched matmul is performed if
        ``m.dim() > 2``, but only last dim reduction is supported
    x : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius matvec result
    """
    return _mobius_matvec(m, x, k, dim=dim)


def _mobius_matvec(m, x, k, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    if dim != -1 or m.dim() == 2:
        mx = torch.tensordot(x, m, dims=([dim], [1]))
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_k(mx_norm / x_norm * artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


# TODO: check if this extends to gyrovector spaces for positive curvature
# TODO: add plot
def mobius_pointwise_mul(w, x, *, k=0, dim=-1):
    r"""
    Computes the generalization for point-wise multiplication in gyrovector
    spaces.

    The Möbius pointwise multiplication is defined as follows

    .. math::

        \operatorname{diag}(w) \otimes_\kappa x = \tan_\kappa\left(
            \frac{\|\operatorname{diag}(w)x\|_2}{x}\tanh^{-1}(\|x\|_2)
        \right)\frac{\|\operatorname{diag}(w)x\|_2}{\|x\|_2}


    Parameters
    ----------
    w : tensor
        weights for multiplication (should be broadcastable to x)
    x : tensor
        point on manifold
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius point-wise mul result
    """
    return _mobius_pointwise_mul(w, x, k, dim=dim)


def _mobius_pointwise_mul(w, x, k, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    wx = w * x
    wx_norm = wx.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_k(wx_norm / x_norm * artan_k(x_norm, k), k) * (wx / wx_norm)
    cond = wx.isclose(wx.new_zeros(0)).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


def mobius_fn_apply_chain(x, *fns, k=0, dim=-1):
    r"""
    Computes the generalization of sequential function application in gyrovector
    spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{log}^\kappa_0` and then the sequence of functions is
    applied to the vector in the tangent space. The resulting tangent vector is
    then mapped back with :math:`\operatorname{exp}^\kappa_0`.

    .. math::

        f^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(f(\operatorname{log}^\kappa_0(y)))

    The definition of mobius function application allows chaining as

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Resulting in

    .. math::

        (f \circ g)^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(
            (f \circ g) (\operatorname{log}^\kappa_0(y))
        )

    Parameters
    ----------
    x : tensor
        point on manifold
    fns : callable[]
        functions to apply
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Apply chain result
    """
    if not fns:
        return x
    else:
        ex = _logmap0(x, k, dim=dim)
        for fn in fns:
            ex = fn(ex)
        y = _expmap0(ex, k, dim=dim)
        return y


def mobius_fn_apply(fn, x, *args, k=0, dim=-1, **kwargs):
    r"""
    Computes the generalization of function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{log}^\kappa_0` and then the function is applied
    to the vector in the tangent space. The resulting tangent vector is then
    mapped back with :math:`\operatorname{exp}^\kappa_0`.

    .. math::

        f^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(f(\operatorname{log}^\kappa_0(y)))

    .. plot:: plots/extended/universal/mobius_sigmoid_apply.py

    Parameters
    ----------
    x : tensor
        point on manifold
    fn : callable
        function to apply
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Result of function in hyperbolic space
    """
    ex = _logmap0(x, k, dim=dim)
    ex = fn(ex, *args, **kwargs)
    y = _expmap0(ex, k, dim=dim)
    return y


def mobiusify(fn):
    r"""
    Wraps a function such that is works in gyrovector spaces.

    Parameters
    ----------
    fn : callable
        function in Euclidean space

    Returns
    -------
    callable
        function working in gyrovector spaces

    Notes
    -----
    New function will accept additional argument ``k`` and ``dim``.
    """

    @functools.wraps(fn)
    def mobius_fn(x, *args, k=0, dim=-1, **kwargs):
        ex = _logmap0(x, k, dim=dim)
        ex = fn(ex, *args, **kwargs)
        y = _expmap0(ex, k, dim=dim)
        return y

    return mobius_fn


def dist2plane(x, p, a, *, k=0, keepdim=False, signed=False, dim=-1):
    r"""
    Computes the geodesic distance from :math:`x` to a hyperplane going through
    :math:`x` with the normal vector :math:`a`.

    The hyperplane is such that its set of points is orthogonal to :math:`a` and
    contains :math:`p`.

    .. plot:: plots/extended/universal/distance2plane.py

    To form an intuition what is a hyperplane in gyrovector spaces, let's first
    consider an Euclidean hyperplane

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\},

    where :math:`a\in \mathbb{R}^n\backslash \{\mathbf{0}\}` and
    :math:`b\in \mathbb{R}^n`.

    This formulation of a hyperplane is hard to generalize,
    therefore we can rewrite :math:`\langle x, a\rangle - b`
    utilizing orthogonal completion.
    Setting any :math:`p` s.t. :math:`b=\langle a, p\rangle` we have

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\}\\
        =H_{a, \langle a, p\rangle} = \tilde{H}_{a, p}\\
        = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - \langle a, p\rangle = 0
        \right\}\\
        =\left\{
            x \in \mathbb{R}^n\;:\;\langle -p + x, a\rangle = 0
        \right\}\\
        = p + \{a\}^\perp

    Naturally we have a set :math:`\{a\}^\perp` with applied :math:`+` operator
    to each element. Generalizing a notion of summation to the gyrovector space
    we replace :math:`+` with :math:`\oplus_\kappa`.

    Next, we should figure out what is :math:`\{a\}^\perp` in the gyrovector
    space.

    First thing that we should acknowledge is that notion of orthogonality is
    defined for vectors in tangent spaces. Let's consider now
    :math:`p\in \mathcal{M}_\kappa^n` and
    :math:`a\in T_p\mathcal{M}_\kappa^n\backslash \{\mathbf{0}\}`.

    Slightly deviating from traditional notation let's write
    :math:`\{a\}_p^\perp` highlighting the tight relationship of
    :math:`a\in T_p\mathcal{M}_\kappa^n\backslash \{\mathbf{0}\}`
    with :math:`p \in \mathcal{M}_\kappa^n`. We then define

    .. math::

        \{a\}_p^\perp := \left\{
            z\in T_p\mathcal{M}_\kappa^n \;:\; \langle z, a\rangle_p = 0
        \right\}

    Recalling that a tangent vector :math:`z` for point :math:`p` yields
    :math:`x = \operatorname{exp}^\kappa_p(z)` we rewrite the above equation as

    .. math::
        \{a\}_p^\perp := \left\{
            x\in \mathcal{M}_\kappa^n \;:\; \langle
            \operatorname{log}_p^\kappa(x), a\rangle_p = 0
        \right\}

    This formulation is something more pleasant to work with.
    Putting all together

    .. math::

        \tilde{H}_{a, p}^\kappa = p + \{a\}^\perp_p\\
        = \left\{
            x \in \mathcal{M}_\kappa^n\;:\;\langle
            \operatorname{log}^\kappa_p(x),
            a\rangle_p = 0
        \right\} \\
        = \left\{
            x \in \mathcal{M}_\kappa^n\;:\;\langle -p \oplus_\kappa x, a\rangle
            = 0
        \right\}

    To compute the distance :math:`d_\kappa(x, \tilde{H}_{a, p}^\kappa)` we find

    .. math::

        d_\kappa(x, \tilde{H}_{a, p}^\kappa)
        =
        \inf_{w\in \tilde{H}_{a, p}^\kappa} d_\kappa(x, w)\\
        =
        \sin^{-1}_\kappa\left\{
            \frac{
            2 |\langle(-p)\oplus_\kappa x, a\rangle|
            }{
            (1+\kappa\|(-p)\oplus_\kappa \|x\|^2_2)\|a\|_2
            }
        \right\}

    Parameters
    ----------
    x : tensor
        point on manifold to compute distance for
    a : tensor
        hyperplane normal vector in tangent space of :math:`p`
    p : tensor
        point on manifold lying on the hyperplane
    k : float|tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    signed : bool
        return signed distance
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        distance to the hyperplane
    """
    return _dist2plane(x, a, p, k, keepdim=keepdim, signed=signed, dim=dim)


def _dist2plane(x, a, p, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
    diff = _mobius_add(-p, x, k, dim=dim)
    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
    sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
    num = 2.0 * sc_diff_a
    denom = ((1 + k * diff_norm2) * a_norm).clamp_min(MIN_NORM)
    return arsin_k(num / denom, k)


def antipode(x, k=0, dim: int = -1):
    r"""
    Computes the antipode of a point :math:`x_1,...,x_n` for :math:`\kappa > 0`.

    Let :math:`x` be a point on some sphere. Then :math:`-x` is its antipode.
    Since we're dealing with stereographic projections, for :math:`sproj(x)` we
    get the antipode :math:`sproj(-x)`. Which is given as follows:

    .. math::

        \text{antipode}(x)
        =
        \frac{1+\kappa\|x\|^2_2}{2\kappa\|x\|^2_2}{}(-x)

    Parameters
    ----------
    x : tensor
        points :math:`x_1,...,x_n` on manifold to compute antipode for
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        antipode
    """
    return _antipode(x, k, dim=dim)


def _antipode(x, k, dim=-1):
    # TODO: add implementation that uses stereographic projections!!!
    # TODO: this one is correct, but it could be more efficient!!!
    v = x / x.norm(p=2, dim=dim, keepdim=True)
    R = 1.0 / k.abs().sqrt()
    pi = 3.141592653589793

    return _geodesic_unit(pi * R, x, v, k, dim=dim)


def gyration(a, b, u, *, k=0, dim=-1):
    r"""
    Computes the gyration of :math:`u` by :math:`[a,b]`.

    The gyration is a special operation of gyrovector spaces. The gyrovector
    space addition operation :math:`\oplus_\kappa` is not associative (as
    mentioned in :func:`mobius_add`), but it is gyroassociative, which means

    .. math::

        u \oplus_\kappa (v \oplus_\kappa w)
        =
        (u\oplus_\kappa v) \oplus_\kappa \operatorname{gyr}[u, v]w,

    where

    .. math::

        \operatorname{gyr}[u, v]w
        =
        \ominus (u \oplus_\kappa v) \oplus (u \oplus_\kappa (v \oplus_\kappa w))

    We can simplify this equation using the explicit formula for the Möbius
    addition [1]. Recall,

    .. math::

        A = - \kappa^2 \langle u, w\rangle \langle v, v\rangle
            - \kappa \langle v, w\rangle
            + 2 \kappa^2 \langle u, v\rangle \langle v, w\rangle\\
        B = - \kappa^2 \langle v, w\rangle \langle u, u\rangle
            + \kappa \langle u, w\rangle\\
        D = 1 - 2 \kappa \langle u, v\rangle
            + \kappa^2 \langle u, u\rangle \langle v, v\rangle\\

        \operatorname{gyr}[u, v]w = w + 2 \frac{A u + B v}{D}.

    Parameters
    ----------
    a : tensor
        first point on manifold
    b : tensor
        second point on manifold
    u : tensor
        vector field for operation
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of automorphism

    References
    ----------
    [1]  A. A. Ungar (2009), A Gyrovector Space Approach to Hyperbolic Geometry
    """
    return _gyration(a, b, u, k, dim=dim)


def _gyration(u, v, w, k, dim: int = -1):
    # non-simplified
    # mupv = -_mobius_add(u, v, K)
    # vpw = _mobius_add(u, w, K)
    # upvpw = _mobius_add(u, vpw, K)
    # return _mobius_add(mupv, upvpw, K)
    # simplified
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    K2 = k ** 2
    a = -K2 * uw * v2 - k * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 + k * uw
    d = 1 - 2 * k * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def parallel_transport(x, y, v, *, k=0, dim=-1):
    r"""
    Computes the parallel transport of :math:`v` from :math:`x` to :math:`y`.

    The parallel transport is essential for adaptive algorithms on Riemannian
    manifolds. For gyrovector spaces the parallel transport is expressed through
    the gyration.

    .. plot:: plots/extended/universal/gyrovector_parallel_transport.py

    To recover parallel transport we first need to study isomorphisms between
    gyrovectors and vectors. The reason is that originally, parallel transport
    is well defined for gyrovectors as

    .. math::

        P_{x\to y}(z) = \operatorname{gyr}[y, -x]z,

    where :math:`x,\:y,\:z \in \mathcal{M}_\kappa^n` and
    :math:`\operatorname{gyr}[a, b]c = \ominus (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa c))`

    But we want to obtain parallel transport for vectors, not for gyrovectors.
    The blessing is the isomorphism mentioned above. This mapping is given by

    .. math::

        U^\kappa_p \: : \: T_p\mathcal{M}_\kappa^n \to \mathbb{G}
        =
        v \mapsto \lambda^\kappa_p v


    Finally, having the points :math:`x,\:y \in \mathcal{M}_\kappa^n` and a
    tangent vector :math:`u\in T_x\mathcal{M}_\kappa^n` we obtain

    .. math::

        P^\kappa_{x\to y}(v)
        =
        (U^\kappa_y)^{-1}\left(\operatorname{gyr}[y, -x] U^\kappa_x(v)\right)\\
        =
        \operatorname{gyr}[y, -x] v \lambda^\kappa_x / \lambda^\kappa_y

    .. plot:: plots/extended/universal/parallel_transport.py


    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector at x to be transported to y
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k, dim=dim)


def _parallel_transport(x, y, u, k, dim: int = -1):
    return (
        _gyration(y, -x, u, k, dim=dim)
        * _lambda_x(x, k, keepdim=True, dim=dim)
        / _lambda_x(y, k, keepdim=True, dim=dim)
    )


def parallel_transport0(y, v, *, k=0, dim=-1):
    r"""
    Computes the parallel transport of :math:`v` from the origin :math:`0` to
    :math:`y`.

    This is just a special case of the parallel transport with the starting
    point at the origin that can be computed more efficiently and more
    numerically stable.

    Parameters
    ----------
    y : tensor
        target point
    v : tensor
        vector to be transported from the origin to y
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0(y, v, k, dim=dim)


def _parallel_transport0(y, v, k, dim: int = -1):
    return v * (1 + k * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)


def parallel_transport0back(x, v, *, k=0, dim: int = -1):
    r"""
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, k=k, dim=dim)


def _parallel_transport0back(x, v, k, dim: int = -1):
    return v / (1 + k * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)


def egrad2rgrad(x, grad, *, k=0, dim=-1):
    r"""
    Converts the Euclidean gradient to the Riemannian gradient in the tangent
    space of :math:`x`.

    .. math::

        \nabla_x = \nabla^E_x / (\lambda_x^\kappa)^2

    Parameters
    ----------
    x : tensor
        point on the manifold
    grad : tensor
        Euclidean gradient for :math:`x`
    k : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in T_x\mathcal{M}_\kappa^n`
    """
    return _egrad2rgrad(x, grad, k, dim=dim)


def _egrad2rgrad(x, grad, k, dim: int = -1):
    return grad / _lambda_x(x, k, keepdim=True, dim=dim) ** 2


# Max: progress for the commit stops here


def sproj(x, k):
    factor = 1.0 / (1.0 + torch.sqrt(k.abs()) * x[:, -1])
    proj = factor[:, None] * x[:, :-1]
    return proj


def inv_sproj(x, k):
    lam_x = _lambda_x(x, k, keepdim=True, dim=-1)
    A = lam_x[:, None] * x
    B = 1.0 / k.abs().sqrt() * (lam_x - 1.0).unsqueeze(dim=-1)
    proj = torch.cat((A, B), dim=-1)
    return proj


def weighted_midpoint(x, a, *, K=1.0, keepdim=False, dim=-1):
    r"""
    Computes the weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights :math:`\alpha_1,...,\alpha_n`.

    The gyromidpoint looks as follows:

    .. plot:: plots/extended/universal/midpoint.py

    The weighted Möbius gyromidpoint is computed as follows

    .. math::

        m_{\kappa}(x_1,\ldots,x_n,\alpha_1,\ldots,\alpha_n)
        =
        \frac{1}{2}
        \otimes_\kappa
        \left(
        \sum_{i=1}^n
        \frac{
        \alpha_i\lambda_{x_i}^\kappa
        }{
        \sum_{j=1}^n\alpha_j(\lambda_{x_j}^\kappa-1)
        }
        x_i
        \right)

    where the weights :math:`\alpha_1,...,\alpha_n` do not necessarily need
    to sum to 1 (only their relative weight matters). Note that this formula
    also requires to choose between the midpoint and its antipode for
    :math:`\kappa > 0`.

    Parameters
    ----------
    x : tensor
        points :math:`x_1,...,x_n` on manifold to compute weighted Möbius
        gyromidpoint for
    a : tensor
        scalar midpoint weights :math:`\alpha_1,...,\alpha_n`
    K : float|tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        weighted Möbius gyromidpoint
    """
    return _weighted_midpoint(x, a, K, keepdim=keepdim, dim=dim)


def _weighted_midpoint(x, w, K, keepdim: bool = False, dim: int = -1):
    lam_x = _lambda_x(x, K, keepdim=False, dim=dim)
    w_times_lam_x = w * lam_x
    denominator = (w_times_lam_x - w).sum()

    # min-clamp denominator
    s = torch.sign(torch.sign(denominator) + 0.1)
    if denominator.abs() < MIN_NORM:
        denominator = s * MIN_NORM
    linear_weights = w_times_lam_x / denominator

    # multiply rows of X by linear weights
    # TODO: incorporate dimension independence in next two lines
    x = x.t()
    rhs = torch.matmul(x, linear_weights).t()
    x = x.t()  # restore
    # TODO: remove dimension appropriately (the specified one)
    if not keepdim:
        rhs = rhs.squeeze()

    # determine midpoint
    midpoint = None
    m = _mobius_scalar_mul(0.5, rhs, K, dim=dim)
    # also compute and compare to antipode of m for positive curvature
    if K > 0:
        m_a = _antipode(m, K, dim=dim)
        # determine whether m or m_a minimizes the sum of distances
        d = _dist(x, m, K, keepdim=keepdim, dim=dim).sum(dim=dim, keepdim=False)
        d_a = _dist(x, m_a, K, keepdim=keepdim, dim=dim).sum(dim=dim, keepdim=False)
        # use midpoint that has smaller sum of squared distances
        midpoint = m if d < d_a else m_a
    else:
        midpoint = m

    return midpoint
