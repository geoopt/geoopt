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
hyperbolic) depending on the value of the negative curvature c. The negative
curvature may be a single number, or a vector equal to the number of rows in 
x (one neg. curvature per row).
"""

def tan_func(x, c):
    c_size = c.shape[-1] if c.dim() > 0 else 1
    c_greater_zero = c > 0
    num_greater_zero = c_greater_zero.sum()
    if num_greater_zero == c_size:
        return tanh(x)
    elif num_greater_zero == 0:
        return torch.tan(x)
    else:
        tanh_reults = tanh(x)
        tan_results = torch.tan(x)
        return torch.where(c_greater_zero, tanh_reults, tan_results)


def arctan_func(x, c):
    c_size = c.shape[-1] if c.dim() > 0 else 1
    c_greater_zero = c > 0
    num_greater_zero = c_greater_zero.sum()
    if num_greater_zero == c_size:
        return artanh(x)
    elif num_greater_zero == 0:
        return torch.atan(x)
    else:
        arctanh_results = artanh(x)
        arctan_results = torch.atan(x)
        return torch.where(c_greater_zero, arctanh_results, arctan_results)


def arcsin_func(x, c):
    c_size = c.shape[-1] if c.dim() > 0 else 1
    c_greater_zero = c > 0
    num_greater_zero = c_greater_zero.sum()
    if num_greater_zero == c_size:
        return arsinh(x)
    elif num_greater_zero == 0:
        return torch.asin(x)
    else:
        arcsinh_results = arsinh(x)
        arcsin_results = torch.asin(x)
        return torch.where(c_greater_zero, arcsinh_results, arcsin_results)


# GYROVECTOR SPACE MATH ########################################################


def project(x, *, c=1.0, dim=-1, eps=None):
    r"""
    Safe projects :math:`x` into the manifold for numerical stability. Only has
    an effect for the Poincaré ball, not for the stereographic projection of the
    sphere.

    Parameters
    ----------
    x : tensor
        point on the manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
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
    return _project(x, c, dim, eps)


def _project(x, c, dim: int = -1, eps: float = None):
    c_greater_zero = c > 0
    num_greater_zero = c_greater_zero.sum()
    # this check is done to improve performance
    # (no projections or norm-checks if c <= 0)
    if num_greater_zero > 0:
        norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
        if eps is None:
            eps = BALL_EPS[x.dtype]
        maxnorm = (1 - eps) / (c.abs().sqrt())
        cond = (norm > maxnorm) * c_greater_zero
        projected = (x / norm) * maxnorm
        return torch.where(cond, projected, x)
    else:
        return x


def lambda_x(x, *, c=1.0, keepdim=False, dim=-1):
    r"""
    Computes the conformal factor :math:`\lambda^c_x` at the point :math:`x` on
    the manifold.

    .. math::

        \lambda^c_x = \frac{1}{1 - c \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        conformal factor
    """
    return _lambda_x(x, c, keepdim=keepdim, dim=dim)


def _lambda_x(x, c, keepdim: bool = False, dim: int = -1):
    return 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(MIN_NORM)


def inner(x, u, v, *, c=1.0, keepdim=False, dim=-1):
    r"""
    Computes the inner product for two vectors :math:`u,v` in the tangent space
    of :math:`x` w.r.t the Riemannian metric of the manifold.

    .. math::

        \langle u, v\rangle_x = (\lambda^c_x)^2 \langle u, v \rangle

    Parameters
    ----------
    x : tensor
        point on the manifold
    u : tensor
        tangent vector to :math:`x` on manifold
    v : tensor
        tangent vector to :math:`x` on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(x, u, v, c, keepdim=keepdim, dim=dim)


def _inner(x, u, v, c, keepdim: bool = False, dim: int = -1):
    return _lambda_x(x, c, keepdim=True, dim=dim) ** 2 * (u * v).sum(
        dim=dim, keepdim=keepdim
    )


def norm(x, u, *, c=1.0, keepdim=False, dim=-1):
    r"""
    Computes the norm of a vectors :math:`u` in the tangent space of :math:`x`
    w.r.t the Riemannian metric of the manifold.

    .. math::

        \|u\|_x = \lambda^c_x \|u\|_2

    Parameters
    ----------
    x : tensor
        point on the manifold
    u : tensor
        tangent vector to :math:`x` on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(x, u, c, keepdim=keepdim, dim=dim)


def _norm(x, u, c, keepdim: bool = False, dim: int = -1):
    return _lambda_x(x, c, keepdim=keepdim, dim=dim) * u.norm(
        dim=dim, keepdim=keepdim, p=2
    )

# TODO: check numerical correctness with Gregor's paper
def mobius_add(x, y, *, c=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector addition.

    .. math::

        x \oplus_c y = \frac{
            (1 + 2 c \langle x, y\rangle + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + 2 c \langle x, y\rangle + c^2 \|x\|^2_2 \|y\|^2_2
        }

    .. plot:: plots/extended/universal/mobius_add.py

    In general this operation is not commutative:

    .. math::

        x \oplus_c y \ne y \oplus_c x

    But in some cases this property holds:

    * zero vector case

    .. math::

        \mathbf{0} \oplus_c x = x \oplus_c \mathbf{0}

    * zero negative curvature case that is same as Euclidean addition

    .. math::

        x \oplus_0 y = y \oplus_0 x

    Another useful property is so called left-cancellation law:

    .. math::

        (-x) \oplus_c (x \oplus_c y) = y

    Parameters
    ----------
    x : tensor
        point on the manifold
    y : tensor
        point on the manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius addition
    """
    return _mobius_add(x, y, c, dim=dim)

# TODO: check numerical correctness with Gregor's paper
def _mobius_add(x, y, c, dim=-1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    # minimize denom (omit c to simplify th notation)
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


def mobius_sub(x, y, *, c=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector subtraction.

    The Möbius subtraction can be represented via the Möbius addition as
    follows:

    .. math::

        x \ominus_c y = x \oplus_c (-y)

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius subtraction
    """
    return _mobius_sub(x, y, c, dim=dim)


def _mobius_sub(x, y, c, dim: int = -1):
    return _mobius_add(x, -y, c, dim=dim)


def mobius_coadd(x, y, *, c=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector coaddition.

    The addition operation :math:`\oplus_c` is neither associative, nor
    commutative. In contrast, the coaddition :math:`\boxplus_c` (or cooperation)
    is an associative operation that is defined as follows.

    .. math::

        a \boxplus_c b = b \boxplus_c a = a\operatorname{gyr}[a, -b]b\\
        = \frac{
            (1 + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + c^2 \|x\|^2_2 \|y\|^2_2
        },

    where :math:`\operatorname{gyr}[a, b]c = \ominus_c (a \oplus b) \oplus_c (a \oplus_c (b \oplus_c c))`

    The following right cancellation property holds

    .. math::

        (a \boxplus_c b) \ominus_c b = a\\
        (a \oplus_c b) \boxminus_c b = a

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius coaddition

    """
    return _mobius_coadd(x, y, c, dim=dim)

# TODO: check numerical stability with Gregor's paper!!!
def _mobius_coadd(x, y, c, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    num = (1 - c * y2) * x + (1 - c * x2) * y
    denom = 1 - c ** 2 * x2 * y2
    # avoid division by zero in this way
    return num / denom.clamp_min(MIN_NORM)


def mobius_cosub(x, y, *, c=1.0, dim=-1):
    r"""
    Computes the Möbius gyrovector cosubtraction.

    The Möbius cosubtraction is defined as follows:

    .. math::

        a \boxminus_c b = a \boxplus_c -b

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius cosubtraction

    """
    return _mobius_cosub(x, y, c, dim=dim)


def _mobius_cosub(x, y, c, dim: int = -1):
    return _mobius_coadd(x, -y, c, dim=dim)


def mobius_scalar_mul(r, x, *, c=1.0, dim=-1):
    r"""
    Computes the Möbius scalar multiplication.

    .. math::

        r \otimes_c x = (1/\sqrt{|c|}) \tanh_c(r\tanh_c^{-1}(\sqrt{|c|}\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_c x = x \oplus_c \dots \oplus_c x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_c x = r_1 \otimes_c x \oplus r_2 \otimes_c x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_c x = r_1 \otimes_c (r_2 \otimes_c x)

    * Monodistributivity

    .. math::

         r \otimes_c (r_1 \otimes x \oplus r_2 \otimes x) = r \otimes_c (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_c x / \|r \otimes_c x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : float|tensor
        scalar for multiplication
    x : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    """
    return _mobius_scalar_mul(r, x, c, dim=dim)


def _mobius_scalar_mul(r, x, c, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_abs_c = torch.abs(c) ** 0.5
    res_c = tan_func(r * arctan_func(sqrt_abs_c * x_norm, c), c) * x / (x_norm * sqrt_abs_c)
    return res_c


def dist(x, y, *, c=1.0, keepdim=False, dim=-1):
    r"""
    Computes the geodesic distance between :math:`x` and :math:`y` on the
    manifold.

    .. math::

        d_c(x, y) = \frac{2}{\sqrt{|c|}}\tanh_c^{-1}(\sqrt{|c|}\|(-x)\oplus_c y\|_2)

    .. plot:: plots/extended/universal/distance.py

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, c, keepdim=keepdim, dim=dim)


def _dist(x, y, c, keepdim: bool = False, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    dist_c = arctan_func(
        sqrt_abs_c * _mobius_add(-x, y, c, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), c
    )
    return dist_c * 2 / sqrt_abs_c


def dist0(x, *, c=1.0, keepdim=False, dim=-1):
    r"""
    Computes geodesic distance to the manifold's origin.

    Parameters
    ----------
    x : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    """
    return _dist0(x, c, keepdim=keepdim, dim=dim)


def _dist0(x, c, keepdim: bool = False, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    dist_c = arctan_func(sqrt_abs_c * x.norm(dim=dim, p=2, keepdim=keepdim), c)
    return dist_c * 2 / sqrt_abs_c


def geodesic(t, x, y, *, c=1.0, dim=-1):
    r"""
    Computes the point on the geodesic (shortest) path connecting :math:`x` and
    :math:`y` at time :math:`x`.

    The path can also be treated as an extension of the line segment to an
    unbounded geodesic that goes through :math:`x` and :math:`y`. The equation
    of the geodesic is given as:

    .. math::

        \gamma_{x\to y}(t) = x \oplus_c t \otimes_c ((-x) \oplus_c y)

    The properties of the geodesic are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Furthermore, the geodesic also satisfies the property of local distance
    minimization:

    .. math::

         d_c(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which
    yields the above formula with :math:`v=1`.

    However, we can always compute the constant speed :math:`v` from the points
    that the particular path connects:

    .. math::

        v = d_c(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_c(x, y)


    Parameters
    ----------
    t : float|tensor
        travelling time
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        point on the geodesic going through x and y
    """
    return _geodesic(t, x, y, c, dim=dim)


def _geodesic(t, x, y, c, dim: int = -1):
    # this is not very numerically stable
    v = _mobius_add(-x, y, c, dim=dim)
    tv = _mobius_scalar_mul(t, v, c, dim=dim)
    gamma_t = _mobius_add(x, tv, c, dim=dim)
    return gamma_t


def expmap(x, u, *, c=1.0, dim=-1):
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

        \operatorname{Exp}^c_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_c \tanh_c(\sqrt{|c|}/2 \|u\|_x) \frac{u}{\sqrt{|c|}\|u\|_2}

    Parameters
    ----------
    x : tensor
        starting point on manifold
    u : tensor
        speed vector in tangent space at x
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, c, dim=dim)


def _expmap(x, u, c, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
            tan_func(sqrt_abs_c / 2 * _lambda_x(x, c, keepdim=True, dim=dim) * u_norm, c)
        * u
        / (sqrt_abs_c * u_norm)
    )
    gamma_1 = _mobius_add(x, second_term, c, dim=dim)
    return gamma_1


def expmap0(u, *, c=1.0, dim=-1):
    r"""
    Computes the exponential map of :math:`u` at the origin :math:`0`.

    .. math::

        \operatorname{Exp}^c_0(u) = \tanh_c(\sqrt{|c|}/2 \|u\|_2) \frac{u}{\sqrt{|c|}\|u\|_2}

    Parameters
    ----------
    u : tensor
        speed vector on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, c, dim=dim)


def _expmap0(u, c, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tan_func(sqrt_abs_c * u_norm, c) * u / (sqrt_abs_c * u_norm)
    return gamma_1


def geodesic_unit(t, x, u, *, c=1.0, dim=-1):
    r"""
    Computes the point on the unit speed geodesic at time :math:`t`, starting
    from :math:`x` with initial direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{x,u}(t) = x\oplus_c \tanh_c(t\sqrt{|c|}/2) \frac{u}{\sqrt{|c|}\|u\|_2}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point on manifold
    u : tensor
        initial direction in tangent space at x
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the point on the unit speed geodesic
    """
    return _geodesic_unit(t, x, u, c, dim=dim)


def _geodesic_unit(t, x, u, c, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = tan_func(sqrt_abs_c / 2 * t, c) * u / (sqrt_abs_c * u_norm)
    gamma_1 = _mobius_add(x, second_term, c, dim=dim)
    return gamma_1


def logmap(x, y, *, c=1.0, dim=-1):
    r"""
    Computes the logarithmic map of :math:`y` at :math:`x`.

    .. math::

        \operatorname{Log}^c_x(y) = \frac{2}{\sqrt{|c|}\lambda_x^c}
        \tanh_c^{-1}(\sqrt{|c|} \|(-x)\oplus_c y\|_2)
        * \frac{(-x)\oplus_c y}{\|(-x)\oplus_c y\|_2}

    The result of the logmap is a vector :math:`u` in the tangent space of
    :math:`x` such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_x M` that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, c, dim=dim)


def _logmap(x, y, c, dim: int = -1):
    sub = _mobius_add(-x, y, c, dim=dim)
    sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(x, c, keepdim=True, dim=dim)
    sqrt_abs_c = torch.abs(c) ** 0.5
    return 2 / sqrt_abs_c / lam * arctan_func(sqrt_abs_c * sub_norm, c) * sub / sub_norm


def logmap0(y, *, c=1.0, dim=-1):
    r"""
    Computes the logarithmic map of :math:`y` at the origin :math:`0`.

    .. math::

        \operatorname{Log}^c_0(y) = \tanh_c^{-1}(\sqrt{|c|}\|y\|_2) \frac{y}{\|y\|_2}

    The result of the logmap at the origin is a vector :math:`u` in the tangent
    space of the origin :math:`0` such that

    .. math::

        y = \operatorname{Exp}^c_0(\operatorname{Log}^c_0(y))

    Parameters
    ----------
    y : tensor
        target point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_0 M` that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, c, dim=dim)


def _logmap0(y, c, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_abs_c * arctan_func(sqrt_abs_c * y_norm, c)


def mobius_matvec(m, x, *, c=1.0, dim=-1):
    r"""
    Computes the generalization of matrix-vector multiplication in gyrovector
    spaces.

    The Möbius matrix vector operation is defined as follows:

    .. math::

        M \otimes_c x = (1/\sqrt{|c|}) \tanh_c\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tanh_c^{-1}(\sqrt{|c|}\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}

    .. plot:: plots/extended/universal/mobius_matvec.py

    Parameters
    ----------
    m : tensor
        matrix for multiplication.
        Batched matmul is performed if ``m.dim() > 2``, but only last dim reduction is supported
    x : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius matvec result
    """
    return _mobius_matvec(m, x, c, dim=dim)


def _mobius_matvec(m, x, c, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_abs_c = torch.abs(c) ** 0.5
    if dim != -1 or m.dim() == 2:
        mx = torch.tensordot(x, m, dims=([dim], [1]))
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_func(mx_norm / x_norm * arctan_func(sqrt_abs_c * x_norm, c), c) * mx / (mx_norm * sqrt_abs_c)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res

# TODO: check if this extends to gyrovector spaces for positive curvature
# TODO: add plot
def mobius_pointwise_mul(w, x, *, c=1.0, dim=-1):
    r"""
    Computes the generalization for point-wise multiplication in gyrovector
    spaces.

    The Möbius pointwise multiplication is defined as follows

    .. math::

        \operatorname{diag}(w) \otimes_c x = (1/\sqrt{|c|}) \tanh_c\left(
            \frac{\|\operatorname{diag}(w)x\|_2}{x}\tanh^{-1}(\sqrt{|c|}\|x\|_2)
        \right)\frac{\|\operatorname{diag}(w)x\|_2}{\|x\|_2}


    Parameters
    ----------
    w : tensor
        weights for multiplication (should be broadcastable to x)
    x : tensor
        point on manifold
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius point-wise mul result
    """
    return _mobius_pointwise_mul(w, x, c, dim=dim)


def _mobius_pointwise_mul(w, x, c, dim: int = -1):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_abs_c = torch.abs(c) ** 0.5
    wx = w * x
    wx_norm = wx.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    res_c = tan_func(wx_norm / x_norm * arctan_func(sqrt_abs_c * x_norm, c), c) * wx / (wx_norm * sqrt_abs_c)
    cond = (wx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


def mobius_fn_apply_chain(x, *fns, c=1.0, dim=-1):
    r"""
    Computes the generalization of sequential function application in gyrovector
    spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{Log}^c_0` and then the sequence of functions is applied
    to the vector in the tangent space. The resulting tangent vector is then mapped
    back with :math:`\operatorname{Exp}^c_0`.

    .. math::

        f^{\otimes_c}(x) = \operatorname{Exp}^c_0(f(\operatorname{Log}^c_0(y)))

    The definition of mobius function application allows chaining as

    .. math::

        y = \operatorname{Exp}^c_0(\operatorname{Log}^c_0(y))

    Resulting in

    .. math::

        (f \circ g)^{\otimes_c}(x) = \operatorname{Exp}^c_0((f \circ g) (\operatorname{Log}^c_0(y)))

    Parameters
    ----------
    x : tensor
        point on manifold
    fns : callable[]
        functions to apply
    c : float|tensor
        negative curvature of manifold (c=-K)
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
        ex = _logmap0(x, c, dim=dim)
        for fn in fns:
            ex = fn(ex)
        y = _expmap0(ex, c, dim=dim)
        return y


def mobius_fn_apply(fn, x, *args, c=1.0, dim=-1, **kwargs):
    r"""
    Computes the generalization of function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{Log}^c_0` and then the function is applied
    to the vector in the tangent space. The resulting tangent vector is then
    mapped back with :math:`\operatorname{Exp}^c_0`.

    .. math::

        f^{\otimes_c}(x) = \operatorname{Exp}^c_0(f(\operatorname{Log}^c_0(y)))

    .. plot:: plots/extended/universal/mobius_sigmoid_apply.py

    Parameters
    ----------
    x : tensor
        point on manifold
    fn : callable
        function to apply
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Result of function in hyperbolic space
    """
    ex = _logmap0(x, c, dim=dim)
    ex = fn(ex, *args, **kwargs)
    y = _expmap0(ex, c, dim=dim)
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
    New function will accept additional argument ``c``.
    """

    @functools.wraps(fn)
    def mobius_fn(x, *args, c=1.0, dim=-1, **kwargs):
        ex = _logmap0(x, c, dim=dim)
        ex = fn(ex, *args, **kwargs)
        y = _expmap0(ex, c, dim=dim)
        return y

    return mobius_fn

# TODO: check if this extends to gyrovector spaces for positive curvature
def dist2plane(x, p, a, *, c=1.0, keepdim=False, signed=False, dim=-1):
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
    Generalizing a notion of summation to the hyperbolic space we replace :math:`+` with :math:`\oplus_c`.

    Next, we should figure out what is :math:`\{a\}^\perp` in the Poincare ball.

    First thing that we should acknowledge is that notion of orthogonality is defined for vectors in tangent spaces.
    Let's consider now :math:`p\in \mathcal{M}_c^n` and :math:`a\in T_p\mathcal{M}_c^n\backslash \{\mathbf{0}\}`.

    Slightly deviating from traditional notation let's write :math:`\{a\}_p^\perp`
    highlighting the tight relationship of :math:`a\in T_p\mathcal{M}_c^n\backslash \{\mathbf{0}\}`
    with :math:`p \in \mathcal{M}_c^n`. We then define

    .. math::

        \{a\}_p^\perp := \left\{
            z\in T_p\mathcal{M}_c^n \;:\; \langle z, a\rangle_p = 0
        \right\}

    Recalling that a tangent vector :math:`z` for point :math:`p` yields :math:`x = \operatorname{Exp}^c_p(z)`
    we rewrite the above equation as

    .. math::
        \{a\}_p^\perp := \left\{
            x\in \mathcal{M}_c^n \;:\; \langle \operatorname{Log}_p^c(x), a\rangle_p = 0
        \right\}

    This formulation is something more pleasant to work with.
    Putting all together

    .. math::

        \tilde{H}_{a, p}^c = p + \{a\}^\perp_p\\
        = \left\{
            x \in \mathcal{M}_c^n\;:\;\langle\operatorname{Log}^c_p(x), a\rangle_p = 0
        \right\} \\
        = \left\{
            x \in \mathcal{M}_c^n\;:\;\langle -p \oplus_c x, a\rangle = 0
        \right\}

    To compute the distance :math:`d_c(x, \tilde{H}_{a, p}^c)` we find

    .. math::

        d_c(x, \tilde{H}_{a, p}^c) = \inf_{w\in \tilde{H}_{a, p}^c} d_c(x, w)\\
        = \frac{1}{\sqrt{c}} \sinh^{-1}\left\{
            \frac{
                2\sqrt{c} |\langle(-p)\oplus_c x, a\rangle|
                }{
                (1-c\|(-p)\oplus_c x\|^2_2)\|a\|_2
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
    c : float|tensor
        negative curvature of manifold (c=-K)
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
    return _dist2plane(x, a, p, c, keepdim=keepdim, signed=signed, dim=dim)


def _dist2plane(x, a, p, c, keepdim: bool = False, signed: bool = False, dim: int = -1):
    sqrt_abs_c = torch.abs(c) ** 0.5
    diff = _mobius_add(-p, x, c, dim=dim)
    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
    sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
    num = 2 * sqrt_abs_c * sc_diff_a
    denom = (1 - c * diff_norm2) * a_norm
    return arcsin_func(num / (denom.clamp_min(MIN_NORM)), c) / sqrt_abs_c


def gyration(a, b, u, *, c=1.0, dim=-1):
    r"""
    Computes the gyration of :math:`u` by :math:`[a,b]`.

    The gyration is a special operation of gyrovector spaces. The gyrovector
    space addition operation :math:`\oplus_c` is not associative (as mentioned
    in :func:`mobius_add`), but it is gyroassociative, which means

    .. math::

        u \oplus_c (v \oplus_c w) = (u\oplus_c v) \oplus_c \operatorname{gyr}[u, v]w,

    where

    .. math::

        \operatorname{gyr}[u, v]w = \ominus (u \oplus_c v) \oplus (u \oplus_c (v \oplus_c w))

    We can simplify this equation using the explicit formula for the Möbius
    addition [1]. Recall,

    .. math::

        A = - c^2 \langle u, w\rangle \langle v, v\rangle + c \langle v, w\rangle +
            2 c^2 \langle u, v\rangle \langle v, w\rangle\\
        B = - c^2 \langle v, w\rangle \langle u, u\rangle - c \langle u, w\rangle\\
        D = 1 + 2 c \langle u, v\rangle + c^2 \langle u, u\rangle \langle v, v\rangle\\

        \operatorname{gyr}[u, v]w = w + 2 \frac{A u + B v}{D}.

    Parameters
    ----------
    a : tensor
        first point on manifold
    b : tensor
        second point on manifold
    u : tensor
        vector field for operation
    c : float|tensor
        negative curvature of manifold (c=-K)
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
    return _gyration(a, b, u, c, dim=dim)


def _gyration(u, v, w, c, dim: int = -1):
    # non-simplified
    # mupv = -_mobius_add(u, v, c)
    # vpw = _mobius_add(u, w, c)
    # upvpw = _mobius_add(u, vpw, c)
    # return _mobius_add(mupv, upvpw, c)
    # simplified
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    c2 = c ** 2
    a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
    b = -c2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + c2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def parallel_transport(x, y, v, *, c=1.0, dim=-1):
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

    where :math:`x,\:y,\:z \in \mathcal{M}_c^n` and
    :math:`\operatorname{gyr}[a, b]c = \ominus (a \oplus_c b) \oplus_c (a \oplus_c (b \oplus_c c))`

    But we want to obtain parallel transport for vectors, not for gyrovectors.
    The blessing is the isomorphism mentioned above. This mapping is given by

    .. math::

        U^c_p \: : \: T_p\mathcal{M}_c^n \to \mathbb{G} = v \mapsto \lambda^c_p v


    Finally, having the points :math:`x,\:y \in \mathcal{M}_c^n` and a
    tangent vector :math:`u\in T_x\mathcal{M}_c^n` we obtain

    .. math::

        P^c_{x\to y}(v) = (U^c_y)^{-1}\left(\operatorname{gyr}[y, -x] U^c_x(v)\right)\\
        = \operatorname{gyr}[y, -x] v \lambda^c_x / \lambda^c_y

    .. plot:: plots/extended/universal/parallel_transport.py


    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector at x to be transported to y
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, c, dim=dim)


def _parallel_transport(x, y, u, c, dim: int = -1):
    return (
        _gyration(y, -x, u, c, dim=dim)
        * _lambda_x(x, c, keepdim=True, dim=dim)
        / _lambda_x(y, c, keepdim=True, dim=dim)
    )


def parallel_transport0(y, v, *, c=1.0, dim=-1):
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
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0(y, v, c, dim=dim)


def _parallel_transport0(y, v, c, dim: int = -1):
    return v * (1 - c * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)


def parallel_transport0back(x, v, *, c=1.0, dim: int = -1):
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
    c : float|tensor
        ball negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, c=c, dim=dim)


def _parallel_transport0back(x, v, c, dim: int = -1):
    return v / (1 - c * x.pow(2).sum(dim=dim, keepdim=True)).clamp_min(MIN_NORM)


def egrad2rgrad(x, grad, *, c=1.0, dim=-1):
    r"""
    Converts the Euclidean gradient to the Riemannian gradient in the tangent
    space of :math:`x`.

    .. math::

        \nabla_x = \nabla^E_x / (\lambda_x^c)^2

    Parameters
    ----------
    x : tensor
        point on the manifold
    grad : tensor
        Euclidean gradient for :math:`x`
    c : float|tensor
        negative curvature of manifold (c=-K)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in T_x\mathcal{M}_x^n`
    """
    return _egrad2rgrad(x, grad, c, dim=dim)


def _egrad2rgrad(x, grad, c, dim: int = -1):
    return grad / _lambda_x(x, c, keepdim=True, dim=dim) ** 2
