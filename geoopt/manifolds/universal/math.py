"""
Poincare ball and stereographic projection of sphere math functions.

The functions for the mathematics in gyrovector spaces are mostly taken from the
papers by Ganea et al. (2018) [1]_, TODO et al. (2019) [2]_ and the book of
Ungar (2005) [3]_.

.. [1] Octavian-Eugen Ganea et al., Hyperbolic Neural Networks, NIPS 2018
.. [2] TODO: add paper!
.. [3] Ungar, Abraham A. Analytic hyperbolic geometry: Mathematical foundations
       and applications. World Scientific, 2005.
"""

import functools
import torch.jit


# NUMERICAL PRECISION ##########################################################


# Clamping safety
MIN_NORM = 1e-15
# Ball epsilon safety border
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# TRIGONOMETRIC FUNCTIONS ######################################################


def tanh(x):
    return x.clamp(-15, 15).tanh()


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
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


# CURVATURE-PARAMETRIZED TRIGONOMETRIC FUNCTIONS ###############################


"""
The following functions select the appropriate trigonometric function (normal or
hyperbolic) depending on the value of the curvature :math:`K`. The curvature may
be a single number, or a vector equal to the number of rows in x (one neg. 
curvature per row).
"""

def tan_func(x, K):
    K_size = K.shape[-1] if K.dim() > 0 else 1
    K_smaller_zero = K < 0
    num_smaller_zero = K_smaller_zero.sum()
    if num_smaller_zero == K_size:
        return tanh(x)
    elif num_smaller_zero == 0:
        return torch.tan(x)
    else:
        tanh_reults = tanh(x)
        tan_results = torch.tan(x)
        return torch.where(K_smaller_zero, tanh_reults, tan_results)


def arctan_func(x, K):
    K_size = K.shape[-1] if K.dim() > 0 else 1
    K_smaller_zero = K < 0
    num_smaller_zero = K_smaller_zero.sum()
    if num_smaller_zero == K_size:
        return artanh(x)
    elif num_smaller_zero == 0:
        return torch.atan(x)
    else:
        arctanh_results = artanh(x)
        arctan_results = torch.atan(x)
        return torch.where(K_smaller_zero, arctanh_results, arctan_results)


def arcsin_func(x, K):
    K_size = K.shape[-1] if K.dim() > 0 else 1
    K_smaller_zero = K < 0
    num_smaller_zero = K_smaller_zero.sum()
    if num_smaller_zero == K_size:
        return arsinh(x)
    elif num_smaller_zero == 0:
        return torch.asin(x)
    else:
        arcsinh_results = arsinh(x)
        arcsin_results = torch.asin(x)
        return torch.where(K_smaller_zero, arcsinh_results, arcsin_results)


# GYROVECTOR SPACE MATH ########################################################


def project(x, *, K=1.0, dim=-1, eps=None):
    r"""
    Safe projects :math:`x` into the manifold for numerical stability. Only has
    an effect for the Poincaré ball, not for the stereographic projection of the
    sphere.

    Parameters
    ----------
    x : tensor
        point on the manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension to compute norm
    eps : float
        stability parameter, uses default for dtype if not provided
        (see BALL_EPS above)

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, K, dim, eps)


def _project(x, K, dim: int = -1, eps: float = None):
    K_smaller_zero = K < 0
    num_smaller_zero = K_smaller_zero.sum()
    # this check is done to improve performance
    # (no projections or norm-checks if K >= 0)
    if num_smaller_zero > 0:
        norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
        if eps is None:
            eps = BALL_EPS[x.dtype]
        maxnorm = (1 - eps) / (K.abs().sqrt())
        cond = (norm > maxnorm) * K_smaller_zero
        projected = (x / norm) * maxnorm
        return torch.where(cond, projected, x)
    else:
        return x


def lambda_x(x, *, K=1.0, keepdim=False, dim=-1):
    r"""
    Computes the conformal factor :math:`\lambda^K_x` at the point :math:`x` on
    the manifold.

    .. math::

        \lambda^K_x = \frac{1}{1 + K \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the manifold
    K : float|tensor
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
    return _lambda_x(x, K, keepdim=keepdim, dim=dim)


def _lambda_x(x, K, keepdim: bool = False, dim: int = -1):
    return 2 / (1 + K * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(MIN_NORM)


def inner(x, u, v, *, K=1.0, keepdim=False, dim=-1):
    r"""
    Computes the inner product for two vectors :math:`u,v` in the tangent space
    of :math:`x` w.r.t the Riemannian metric of the manifold.

    .. math::

        \langle u, v\rangle_x = (\lambda^K_x)^2 \langle u, v \rangle

    Parameters
    ----------
    x : tensor
        point on the manifold
    u : tensor
        tangent vector to :math:`x` on manifold
    v : tensor
        tangent vector to :math:`x` on manifold
    K : float|tensor
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
    return _inner(x, u, v, K, keepdim=keepdim, dim=dim)


def _inner(x, u, v, K, keepdim: bool = False, dim: int = -1):
    return _lambda_x(x, K, keepdim=True, dim=dim) ** 2 * (u * v).sum(
        dim=dim, keepdim=keepdim
    )


def norm(x, u, *, K=1.0, keepdim=False, dim=-1):
    r"""
    Computes the norm of a vectors :math:`u` in the tangent space of :math:`x`
    w.r.t the Riemannian metric of the manifold.

    .. math::

        \|u\|_x = \lambda^K_x \|u\|_2

    Parameters
    ----------
    x : tensor
        point on the manifold
    u : tensor
        tangent vector to :math:`x` on manifold
    K : float|tensor
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
    return _norm(x, u, K, keepdim=keepdim, dim=dim)


def _norm(x, u, K, keepdim: bool = False, dim: int = -1):
    return _lambda_x(x, K, keepdim=keepdim, dim=dim) * u.norm(
        dim=dim, keepdim=keepdim, p=2
    )

# TODO: check numerical correctness with Gregor's paper
def mobius_add(x, y, *, K=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector addition.

    .. math::

        x \oplus_K y = \frac{
            (1 - 2 K \langle x, y\rangle - K \|y\|^2_2) x + (1 + K \|x\|_2^2) y
            }{
            1 - 2 K \langle x, y\rangle + K^2 \|x\|^2_2 \|y\|^2_2
        }

    .. plot:: plots/extended/universal/mobius_add.py

    In general this operation is not commutative:

    .. math::

        x \oplus_K y \ne y \oplus_K x

    But in some cases this property holds:

    * zero vector case

    .. math::

        \mathbf{0} \oplus_K x = x \oplus_K \mathbf{0}

    * zero negative curvature case that is same as Euclidean addition

    .. math::

        x \oplus_0 y = y \oplus_0 x

    Another useful property is so called left-cancellation law:

    .. math::

        (-x) \oplus_K (x \oplus_K y) = y

    Parameters
    ----------
    x : tensor
        point on the manifold
    y : tensor
        point on the manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius addition
    """
    return _mobius_add(x, y, K, dim=dim)

# TODO: check numerical correctness with Gregor's paper
def _mobius_add(x, y, K, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * K * xy - K * y2) * x + (1 + K * x2) * y
    denom = 1 - 2 * K * xy + K ** 2 * x2 * y2
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


def mobius_sub(x, y, *, K=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector subtraction.

    The Möbius subtraction can be represented via the Möbius addition as
    follows:

    .. math::

        x \ominus_K y = x \oplus_K (-y)

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius subtraction
    """
    return _mobius_sub(x, y, K, dim=dim)


def _mobius_sub(x, y, K, dim: int = -1):
    return _mobius_add(x, -y, K, dim=dim)


def mobius_coadd(x, y, *, K=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector coaddition.

    The addition operation :math:`\oplus_K` is neither associative, nor
    commutative. In contrast, the coaddition :math:`\boxplus_K` (or cooperation)
    is an associative operation that is defined as follows.

    .. math::

        a \boxplus_K b = b \boxplus_K a = a\operatorname{gyr}[a, -b]b\\
        = \frac{
            (1 + K \|y\|^2_2) x + (1 + K \|x\|_2^2) y
            }{
            1 + K^2 \|x\|^2_2 \|y\|^2_2
        },

    where :math:`\operatorname{gyr}[a, b]v = \ominus_K (a \oplus_K b) \oplus_K (a \oplus_K (b \oplus_K v))`

    The following right cancellation property holds

    .. math::

        (a \boxplus_K b) \ominus_K b = a\\
        (a \oplus_K b) \boxminus_K b = a

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius coaddition

    """
    return _mobius_coadd(x, y, K, dim=dim)

# TODO: check numerical stability with Gregor's paper!!!
def _mobius_coadd(x, y, K, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    num = (1 + K * y2) * x + (1 + K * x2) * y
    denom = 1 - K ** 2 * x2 * y2
    # avoid division by zero in this way
    return num / denom.clamp_min(MIN_NORM)


def mobius_cosub(x, y, *, K=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector cosubtraction.

    The Möbius cosubtraction is defined as follows:

    .. math::

        a \boxminus_K b = a \boxplus_K -b

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius cosubtraction

    """
    return _mobius_cosub(x, y, K, dim=dim)


def _mobius_cosub(x, y, K, dim: int = -1):
    return _mobius_coadd(x, -y, K, dim=dim)


def mobius_scalar_mul(r, x, *, K=1.0, dim=-1):
    r"""
    Computes the Möbius scalar multiplication.

    .. math::

        r \otimes_K x = (1/\sqrt{|K|}) \tanh_K(r\tanh_K^{-1}(\sqrt{|K|}\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_K x = x \oplus_K \dots \oplus_K x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_K x = r_1 \otimes_K x \oplus r_2 \otimes_K x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_K x = r_1 \otimes_K (r_2 \otimes_K x)

    * Monodistributivity

    .. math::

         r \otimes_K (r_1 \otimes x \oplus r_2 \otimes x) = r \otimes_K (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_K x / \|r \otimes_K x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : float|tensor
        scalar for multiplication
    x : tensor
        point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    """
    return _mobius_scalar_mul(r, x, K, dim=dim)


def _mobius_scalar_mul(r, x, K, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_abs_K = torch.abs(K) ** 0.5
    res_c = tan_func(r * arctan_func(sqrt_abs_K * x_norm, K), K) * x / (x_norm * sqrt_abs_K)
    return res_c


def dist(x, y, *, K=1.0, keepdim=False, dim=-1):
    r"""
    Computes the geodesic distance between :math:`x` and :math:`y` on the
    manifold.

    .. math::

        d_K(x, y) = \frac{2}{\sqrt{|K|}}\tanh_K^{-1}(\sqrt{|K|}\|(-x)\oplus_K y\|_2)

    .. plot:: plots/extended/universal/distance.py

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    K : float|tensor
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
    return _dist(x, y, K, keepdim=keepdim, dim=dim)


def _dist(x, y, K, keepdim: bool = False, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    dist_K = arctan_func(
        sqrt_abs_K *
        _mobius_add(-x, y, K, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), K
    )
    return dist_K * 2 / sqrt_abs_K


def dist0(x, *, K=1.0, keepdim=False, dim=-1):
    r"""
    Computes geodesic distance to the manifold's origin.

    Parameters
    ----------
    x : tensor
        point on manifold
    K : float|tensor
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
    return _dist0(x, K, keepdim=keepdim, dim=dim)


def _dist0(x, K, keepdim: bool = False, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    dist_K = arctan_func(sqrt_abs_K * x.norm(dim=dim, p=2, keepdim=keepdim), K)
    return dist_K * 2 / sqrt_abs_K


def geodesic(t, x, y, *, K=1.0, dim=-1):
    r"""
    Computes the point on the geodesic (shortest) path connecting :math:`x` and
    :math:`y` at time :math:`x`.

    The path can also be treated as an extension of the line segment to an
    unbounded geodesic that goes through :math:`x` and :math:`y`. The equation
    of the geodesic is given as:

    .. math::

        \gamma_{x\to y}(t) = x \oplus_K t \otimes_K ((-x) \oplus_K y)

    The properties of the geodesic are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Furthermore, the geodesic also satisfies the property of local distance
    minimization:

    .. math::

         d_K(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which
    yields the above formula with :math:`v=1`.

    However, we can always compute the constant speed :math:`v` from the points
    that the particular path connects:

    .. math::

        v = d_K(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_K(x, y)


    Parameters
    ----------
    t : float|tensor
        travelling time
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        point on the geodesic going through x and y
    """
    return _geodesic(t, x, y, K, dim=dim)


def _geodesic(t, x, y, K, dim: int = -1):
    # this is not very numerically stable
    v = _mobius_add(-x, y, K, dim=dim)
    tv = _mobius_scalar_mul(t, v, K, dim=dim)
    gamma_t = _mobius_add(x, tv, K, dim=dim)
    return gamma_t


def expmap(x, u, *, K=1.0, dim=-1):
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

        \operatorname{Exp}^K_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_K \tanh_K(\sqrt{|K|}/2 \|u\|_x) \frac{u}{\sqrt{|K|}\|u\|_2}

    Parameters
    ----------
    x : tensor
        starting point on manifold
    u : tensor
        speed vector in tangent space at x
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, K, dim=dim)


def _expmap(x, u, K, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
            tan_func(sqrt_abs_K / 2 * _lambda_x(x, K, keepdim=True, dim=dim) * u_norm, K)
        * u
        / (sqrt_abs_K * u_norm)
    )
    gamma_1 = _mobius_add(x, second_term, K, dim=dim)
    return gamma_1


def expmap0(u, *, K=1.0, dim=-1):
    r"""
    Computes the exponential map of :math:`u` at the origin :math:`0`.

    .. math::

        \operatorname{Exp}^K_0(u) = \tanh_K(\sqrt{|K|}/2 \|u\|_2) \frac{u}{\sqrt{|K|}\|u\|_2}

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
    return _expmap0(u, K, dim=dim)


def _expmap0(u, K, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tan_func(sqrt_abs_K * u_norm, K) * u / (sqrt_abs_K * u_norm)
    return gamma_1


def geodesic_unit(t, x, u, *, K=1.0, dim=-1):
    r"""
    Computes the point on the unit speed geodesic at time :math:`t`, starting
    from :math:`x` with initial direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{x,u}(t) = x\oplus_K \tanh_K(t\sqrt{|K|}/2) \frac{u}{\sqrt{|K|}\|u\|_2}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point on manifold
    u : tensor
        initial direction in tangent space at x
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the point on the unit speed geodesic
    """
    return _geodesic_unit(t, x, u, K, dim=dim)


def _geodesic_unit(t, x, u, K, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = tan_func(sqrt_abs_K / 2 * t, K) * u / (sqrt_abs_K * u_norm)
    gamma_1 = _mobius_add(x, second_term, K, dim=dim)
    return gamma_1


def logmap(x, y, *, K=1.0, dim=-1):
    r"""
    Computes the logarithmic map of :math:`y` at :math:`x`.

    .. math::

        \operatorname{Log}^K_x(y) = \frac{2}{\sqrt{|K|}\lambda_x^K}
        \tanh_K^{-1}(\sqrt{|K|} \|(-x)\oplus_K y\|_2)
        * \frac{(-x)\oplus_K y}{\|(-x)\oplus_K y\|_2}

    The result of the logmap is a vector :math:`u` in the tangent space of
    :math:`x` such that

    .. math::

        y = \operatorname{Exp}^K_x(\operatorname{Log}^K_x(y))


    Parameters
    ----------
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_x M` that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, K, dim=dim)


def _logmap(x, y, K, dim: int = -1):
    sub = _mobius_add(-x, y, K, dim=dim)
    sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(x, K, keepdim=True, dim=dim)
    sqrt_abs_K = torch.abs(K) ** 0.5
    return 2 / sqrt_abs_K / lam * arctan_func(sqrt_abs_K * sub_norm, K) * sub / sub_norm


def logmap0(y, *, K=1.0, dim=-1):
    r"""
    Computes the logarithmic map of :math:`y` at the origin :math:`0`.

    .. math::

        \operatorname{Log}^K_0(y) = \tanh_K^{-1}(\sqrt{|K|}\|y\|_2) \frac{y}{\|y\|_2}

    The result of the logmap at the origin is a vector :math:`u` in the tangent
    space of the origin :math:`0` such that

    .. math::

        y = \operatorname{Exp}^K_0(\operatorname{Log}^K_0(y))

    Parameters
    ----------
    y : tensor
        target point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_0 M` that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, K, dim=dim)


def _logmap0(y, K, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_abs_K * arctan_func(sqrt_abs_K * y_norm, K)


def mobius_matvec(m, x, *, K=1.0, dim=-1):
    r"""
    Computes the generalization of matrix-vector multiplication in gyrovector
    spaces.

    The Möbius matrix vector operation is defined as follows:

    .. math::

        M \otimes_K x = (1/\sqrt{|K|}) \tanh_K\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tanh_K^{-1}(\sqrt{|K|}\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}

    .. plot:: plots/extended/universal/mobius_matvec.py

    Parameters
    ----------
    m : tensor
        matrix for multiplication.
        Batched matmul is performed if ``m.dim() > 2``, but only last dim reduction is supported
    x : tensor
        point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius matvec result
    """
    return _mobius_matvec(m, x, K, dim=dim)


def _mobius_matvec(m, x, K, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_abs_K = torch.abs(K) ** 0.5
    if dim != -1 or m.dim() == 2:
        mx = torch.tensordot(x, m, dims=([dim], [1]))
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_func(mx_norm / x_norm * arctan_func(sqrt_abs_K * x_norm, K), K) * mx / (mx_norm * sqrt_abs_K)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res

# TODO: check if this extends to gyrovector spaces for positive curvature
# TODO: add plot
def mobius_pointwise_mul(w, x, *, K=1.0, dim=-1):
    r"""
    Computes the generalization for point-wise multiplication in gyrovector
    spaces.

    The Möbius pointwise multiplication is defined as follows

    .. math::

        \operatorname{diag}(w) \otimes_K x = (1/\sqrt{|K|}) \tanh_K\left(
            \frac{\|\operatorname{diag}(w)x\|_2}{x}\tanh^{-1}(\sqrt{|K|}\|x\|_2)
        \right)\frac{\|\operatorname{diag}(w)x\|_2}{\|x\|_2}


    Parameters
    ----------
    w : tensor
        weights for multiplication (should be broadcastable to x)
    x : tensor
        point on manifold
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius point-wise mul result
    """
    return _mobius_pointwise_mul(w, x, K, dim=dim)


def _mobius_pointwise_mul(w, x, K, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_abs_K = torch.abs(K) ** 0.5
    wx = w * x
    wx_norm = wx.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_func(wx_norm / x_norm * arctan_func(sqrt_abs_K * x_norm, K), K) * wx / (wx_norm * sqrt_abs_K)
    cond = (wx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


def mobius_fn_apply_chain(x, *fns, K=1.0, dim=-1):
    r"""
    Computes the generalization of sequential function application in gyrovector
    spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{Log}^K_0` and then the sequence of functions is applied
    to the vector in the tangent space. The resulting tangent vector is then mapped
    back with :math:`\operatorname{Exp}^K_0`.

    .. math::

        f^{\otimes_K}(x) = \operatorname{Exp}^K_0(f(\operatorname{Log}^K_0(y)))

    The definition of mobius function application allows chaining as

    .. math::

        y = \operatorname{Exp}^K_0(\operatorname{Log}^K_0(y))

    Resulting in

    .. math::

        (f \circ g)^{\otimes_K}(x) = \operatorname{Exp}^K_0((f \circ g) (\operatorname{Log}^K_0(y)))

    Parameters
    ----------
    x : tensor
        point on manifold
    fns : callable[]
        functions to apply
    K : float|tensor
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
        ex = _logmap0(x, K, dim=dim)
        for fn in fns:
            ex = fn(ex)
        y = _expmap0(ex, K, dim=dim)
        return y


def mobius_fn_apply(fn, x, *args, K=1.0, dim=-1, **kwargs):
    r"""
    Computes the generalization of function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{Log}^K_0` and then the function is applied
    to the vector in the tangent space. The resulting tangent vector is then
    mapped back with :math:`\operatorname{Exp}^K_0`.

    .. math::

        f^{\otimes_K}(x) = \operatorname{Exp}^K_0(f(\operatorname{Log}^K_0(y)))

    .. plot:: plots/extended/universal/mobius_sigmoid_apply.py

    Parameters
    ----------
    x : tensor
        point on manifold
    fn : callable
        function to apply
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Result of function in hyperbolic space
    """
    ex = _logmap0(x, K, dim=dim)
    ex = fn(ex, *args, **kwargs)
    y = _expmap0(ex, K, dim=dim)
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
    New function will accept additional argument ``K``.
    """

    @functools.wraps(fn)
    def mobius_fn(x, *args, K=1.0, dim=-1, **kwargs):
        ex = _logmap0(x, K, dim=dim)
        ex = fn(ex, *args, **kwargs)
        y = _expmap0(ex, K, dim=dim)
        return y

    return mobius_fn

# TODO: check if this extends to gyrovector spaces for positive curvature
def dist2plane(x, p, a, *, K=1.0, keepdim=False, signed=False, dim=-1):
    r"""
    Computes the geodesic distance from :math:`x` to a hyperplane going through
    :math:`x` with the normal vector :math:`a`.

    The hyperplane is such that its set of points is orthogonal to :math:`a` and
    contains :math:`p`.

    .. plot:: plots/extended/universal/distance2plane.py

    To form an intuition what is a hyperbolic hyperplane, let's first consider Euclidean hyperplane

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\},

    where :math:`a\in \mathbb{R}^n\backslash \{\mathbf{0}\}` and :math:`b\in \mathbb{R}^n`.

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

    Naturally we have a set :math:`\{a\}^\perp` with applied :math:`+` operator to each element.
    Generalizing a notion of summation to the hyperbolic space we replace :math:`+` with :math:`\oplus_K`.

    Next, we should figure out what is :math:`\{a\}^\perp` in the Poincare ball.

    First thing that we should acknowledge is that notion of orthogonality is defined for vectors in tangent spaces.
    Let's consider now :math:`p\in \mathcal{M}_K^n` and :math:`a\in T_p\mathcal{M}_K^n\backslash \{\mathbf{0}\}`.

    Slightly deviating from traditional notation let's write :math:`\{a\}_p^\perp`
    highlighting the tight relationship of :math:`a\in T_p\mathcal{M}_K^n\backslash \{\mathbf{0}\}`
    with :math:`p \in \mathcal{M}_K^n`. We then define

    .. math::

        \{a\}_p^\perp := \left\{
            z\in T_p\mathcal{M}_K^n \;:\; \langle z, a\rangle_p = 0
        \right\}

    Recalling that a tangent vector :math:`z` for point :math:`p` yields :math:`x = \operatorname{Exp}^K_p(z)`
    we rewrite the above equation as

    .. math::
        \{a\}_p^\perp := \left\{
            x\in \mathcal{M}_K^n \;:\; \langle \operatorname{Log}_p^K(x), a\rangle_p = 0
        \right\}

    This formulation is something more pleasant to work with.
    Putting all together

    .. math::

        \tilde{H}_{a, p}^K = p + \{a\}^\perp_p\\
        = \left\{
            x \in \mathcal{M}_K^n\;:\;\langle\operatorname{Log}^K_p(x), a\rangle_p = 0
        \right\} \\
        = \left\{
            x \in \mathcal{M}_K^n\;:\;\langle -p \oplus_K x, a\rangle = 0
        \right\}

    To compute the distance :math:`d_K(x, \tilde{H}_{a, p}^K)` we find

    .. math::

        d_K(x, \tilde{H}_{a, p}^K) = \inf_{w\in \tilde{H}_{a, p}^K} d_K(x, w)\\
        = \frac{1}{\sqrt{|K|}} \sinh^{-1}_K\left\{
            \frac{
                2\sqrt{|K|} |\langle(-p)\oplus_K x, a\rangle|
                }{
                (1+K\|(-p)\oplus_K x\|^2_2)\|a\|_2
            }
        \right\}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    a : tensor
        vector on tangent space of :math:`p`
    p : tensor
        point on Poincare ball lying on the hyperplane
    K : float|tensor
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
    return _dist2plane(x, a, p, K, keepdim=keepdim, signed=signed, dim=dim)


def _dist2plane(x, a, p, K, keepdim: bool = False, signed: bool = False, dim: int = -1):
    sqrt_abs_K = torch.abs(K) ** 0.5
    diff = _mobius_add(-p, x, K, dim=dim)
    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
    sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
    num = 2 * sqrt_abs_K * sc_diff_a
    denom = (1 + K * diff_norm2) * a_norm
    return arcsin_func(num / (denom.clamp_min(MIN_NORM)), K) / sqrt_abs_K


def gyration(a, b, u, *, K=1.0, dim=-1):
    r"""
    Computes the gyration of :math:`u` by :math:`[a,b]`.

    The gyration is a special operation of gyrovector spaces. The gyrovector
    space addition operation :math:`\oplus_K` is not associative (as mentioned
    in :func:`mobius_add`), but it is gyroassociative, which means

    .. math::

        u \oplus_K (v \oplus_K w) = (u\oplus_K v) \oplus_K \operatorname{gyr}[u, v]w,

    where

    .. math::

        \operatorname{gyr}[u, v]w = \ominus (u \oplus_K v) \oplus (u \oplus_K (v \oplus_K w))

    We can simplify this equation using the explicit formula for the Möbius
    addition [1]. Recall,

    .. math::

        A = - K^2 \langle u, w\rangle \langle v, v\rangle - K \langle v, w\rangle +
            2 K^2 \langle u, v\rangle \langle v, w\rangle\\
        B = - K^2 \langle v, w\rangle \langle u, u\rangle + K \langle u, w\rangle\\
        D = 1 - 2 K \langle u, v\rangle + K^2 \langle u, u\rangle \langle v, v\rangle\\

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
    return _gyration(a, b, u, K, dim=dim)


def _gyration(u, v, w, K, dim: int = -1):
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
    K2 = K ** 2
    a = -K2 * uw * v2 - K * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 + K * uw
    d = 1 - 2 * K * uv + K2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def parallel_transport(x, y, v, *, K=1.0, dim=-1):
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

    where :math:`x,\:y,\:z \in \mathcal{M}_K^n` and
    :math:`\operatorname{gyr}[a, b]c = \ominus (a \oplus_K b) \oplus_K (a \oplus_K (b \oplus_K c))`

    But we want to obtain parallel transport for vectors, not for gyrovectors.
    The blessing is the isomorphism mentioned above. This mapping is given by

    .. math::

        U^K_p \: : \: T_p\mathcal{M}_K^n \to \mathbb{G} = v \mapsto \lambda^K_p v


    Finally, having the points :math:`x,\:y \in \mathcal{M}_K^n` and a
    tangent vector :math:`u\in T_x\mathcal{M}_K^n` we obtain

    .. math::

        P^K_{x\to y}(v) = (U^K_y)^{-1}\left(\operatorname{gyr}[y, -x] U^K_x(v)\right)\\
        = \operatorname{gyr}[y, -x] v \lambda^K_x / \lambda^K_y

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
    return _parallel_transport(x, y, v, K, dim=dim)


def _parallel_transport(x, y, u, K, dim: int = -1):
    return (
        _gyration(y, -x, u, K, dim=dim)
        * _lambda_x(x, K, keepdim=True, dim=dim)
        / _lambda_x(y, K, keepdim=True, dim=dim)
    )


def parallel_transport0(y, v, *, K=1.0, dim=-1):
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
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0(y, v, K, dim=dim)


def _parallel_transport0(y, v, K, dim: int = -1):
    return v * (1 + K * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)


def parallel_transport0back(x, v, *, K=1.0, dim: int = -1):
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
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, K=K, dim=dim)


def _parallel_transport0back(x, v, K, dim: int = -1):
    return v / (1 + K * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)


def egrad2rgrad(x, grad, *, K=1.0, dim=-1):
    r"""
    Converts the Euclidean gradient to the Riemannian gradient in the tangent
    space of :math:`x`.

    .. math::

        \nabla_x = \nabla^E_x / (\lambda_x^K)^2

    Parameters
    ----------
    x : tensor
        point on the manifold
    grad : tensor
        Euclidean gradient for :math:`x`
    K : float|tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in T_x\mathcal{M}_K^n`
    """
    return _egrad2rgrad(x, grad, K, dim=dim)


def _egrad2rgrad(x, grad, K, dim: int = -1):
    return grad / _lambda_x(x, K, keepdim=True, dim=dim) ** 2
