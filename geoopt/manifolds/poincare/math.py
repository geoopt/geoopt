"""
Functions for math on Poincare ball model. Most of this is taken from
a well written paper by Octavian-Eugen Ganea (2018) [1]_


.. [1] Hyperbolic Neural Networks, NIPS2018
        https://arxiv.org/abs/1805.09112
"""

import functools
import torch
import torch.jit


@torch.jit.script
def tanh(x):  # pragma: no cover
    return x.clamp(-15, 15).tanh()


# noinspection PyTypeChecker,PyUnresolvedReferences
@torch.jit.script
def artanh(x):  # pragma: no cover
    x = x.clamp(-1 + 1e-15, 1 - 1e-15)
    res = 0.5 * (torch.log(1 + x) - torch.log(1 - x))
    return res


@torch.jit.script
def arsinh(x):  # pragma: no cover
    return torch.log(x + torch.sqrt(1 + x ** 2))


@torch.jit.script
def arcosh(x):  # pragma: no cover
    x = x.clamp(-1 + 1e-15, 1 - 1e-15)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


def project(x, *, c=1.0):
    r"""
    Safe projection on the manifold for numerical stability. This was mentioned in [1]_

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        projected vector on the manifold

    References
    ----------
    .. [1] Hyperbolic Neural Networks, NIPS2018
        https://arxiv.org/abs/1805.09112
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _project(x, c)


@torch.jit.script
def _project(x, c):  # pragma: no cover
    norm = x.norm(dim=-1, keepdim=True, p=2)
    if x.dtype == torch.float64:
        maxnorm = (1 - 1e-5) / (c ** 0.5)
    else:
        maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x, *, c=1.0, keepdim=False):
    r"""
    Compute the conformal factor :math:`\lambda^c_x` for a point on the ball

    .. math::

        \lambda^c_x = \frac{1}{1 - c \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        conformal factor
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _lambda_x(x, c, keepdim=keepdim)


@torch.jit.script
def _lambda_x(x, c, keepdim: bool = False):  # pragma: no cover
    return 2 / (1 - c * x.pow(2).sum(-1, keepdim=keepdim))


def inner(x, u, v, *, c=1.0, keepdim=False):
    r"""
    Compute inner product for two vectors on the tangent space w.r.t Riemannian metric on the Poincare ball

    .. math::

        \langle u, v\rangle_x = (\lambda^c_x)^2 \langle u, v \rangle

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on poincare ball
    v : tensor
        tangent vector to :math:`x` on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        inner product
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _inner(x, u, v, c, keepdim=keepdim)


@torch.jit.script
def _inner(x, u, v, c, keepdim: bool = False):  # pragma: no cover
    return _lambda_x(x, c, keepdim=True) ** 2 * (u * v).sum(-1, keepdim=keepdim)


def norm(x, u, *, c=1.0, keepdim=False):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Poincare ball

    .. math::

        \|u\|_x = \lambda^c_x \|u\|_2

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        norm of vector
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _norm(x, u, c, keepdim=keepdim)


@torch.jit.script
def _norm(x, u, c, keepdim: bool = False):  # pragma: no cover
    return _lambda_x(x, c, keepdim=keepdim) * u.norm(dim=-1, keepdim=keepdim, p=2)


def mobius_add(x, y, *, c=1.0):
    r"""
    Mobius addition is a special operation in a hyperbolic space.

    .. math::

        x \oplus_c y = \frac{
            (1 + 2 c \langle x, y\rangle + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + 2 c \langle x, y\rangle + c^2 \|x\|^2_2 \|y\|^2_2
        }

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

    Another usefull property is so called left-cancellation law:

    .. math::

        (-x) \oplus_c (x \oplus_c y) = y

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    y : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        the result of mobius addition
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)


@torch.jit.script
def _mobius_add(x, y, c):  # pragma: no cover
    y = y + 1e-15
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    # avoid division by zero in this way
    return num / (denom + 1e-15)


def mobius_sub(x, y, *, c=1.0):
    r"""
    Mobius substraction that can be represented via Mobius addition as follows:

    .. math::

        x \ominus_c y = x \oplus_c (-y)

    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        the result of mobius substraction
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _mobius_sub(x, y, c)


@torch.jit.script
def _mobius_sub(x, y, c):  # pragma: no cover
    return _mobius_add(x, -y, c)


def mobius_scalar_mul(r, x, *, c=1.0):
    r"""
    Left scalar multiplication on the Poincare ball

    .. math::

        r \otimes_c x = (1/\sqrt{c}) \tanh(r\tanh^{-1}(\sqrt{c}\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to euclidean

    * `n-addition` property

    .. math::

         r \otimes_c x = x \oplus_c \dots \oplus_c x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_c x = r_1 \otimes_c x \oplus r_2 \otimes_c x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_c x = r_1 \otimes_c (r_2 \otimes_c x)

    * Scaling property

    .. math::

        |r| \otimes_c x / \|r \otimes_c x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : float|tensor
        scalar for multiplication
    x : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        the result of mobius scalar multiplication
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    if not isinstance(r, torch.Tensor):
        r = torch.as_tensor(r).type_as(x)
    return _mobius_scalar_mul(r, x, c)


@torch.jit.script
def _mobius_scalar_mul(r, x, c):  # pragma: no cover
    x = x + 1e-15
    x_norm = x.norm(dim=-1, keepdim=True, p=2)
    sqrt_c = c ** 0.5
    res_c = tanh(r * artanh(sqrt_c * x_norm)) * x / (x_norm * sqrt_c)
    return _project(res_c, c)


def dist(x, y, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball

    .. math::

        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|(-x)\oplus_c y\|_2)

    .. plot:: plots/extended/poincare/distance.py

    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)


@torch.jit.script
def _dist(x, y, c, keepdim: bool = False):  # pragma: no cover
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def dist0(x, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball to zero

    Parameters
    ----------
    x : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _dist0(x, c, keepdim=keepdim)


@torch.jit.script
def _dist0(x, c, keepdim: bool = False):  # pragma: no cover
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * x.norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def project_tangent(x, u, *, c):
    r"""
    Project tangent vector to reasonable values that do not exceed
    maximum allowed (vector norm allowing to travel to the opposite pole)

    .. math::

        \operatorname{maxnorm}_x = d_{c}(\operatorname{proj}(-\infty), \operatorname{proj}(\infty)) / \lambda_x^c

    Parameters
    ----------
    x : tensor
        point on poincare ball
    u : tensor
        tangent vector
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        same tangent vector with reasonable values
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _project_tangent(x, u, c)


@torch.jit.script
def _project_tangent(x, u, c):  # pragma: no cover
    # get the almost infinite vecotor estimate
    # this is the norm of travel vector to the opposite pole
    dim = x.size(-1)
    p = torch.ones((dim,), dtype=c.dtype, device=c.device)
    p = p / dim ** 0.5 / (c ** 0.5)
    p = _project(p, c)
    # normalize its length based on x
    maxnorm = _dist(p, -p, c, keepdim=True) / _lambda_x(x, c, keepdim=True)
    norm = u.norm(dim=-1, keepdim=True, p=2)
    cond = norm > maxnorm
    projected = u / norm * maxnorm
    return torch.where(cond, projected, u)


def geodesic(t, x, y, *, c=1.0):
    r"""
    Geodesic (the shortest) path connecting :math:`x` and :math:`y`.
    The path can be treated as and extension of a line segment between
    points but in a Riemannian manifold. In Poincare ball model, the path
    is expressed using Mobius addition and scalar multiplication:

    .. math::

        \gamma_{x\to y}(t) = x \oplus_c r \otimes_c ((-x) \oplus_c y)

    The required properties of this path are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Moreover, as geodesic path is not only the shortest path connecting points and Poincare ball.
    This definition also requires local distance minimization and thus another property appears:

    .. math::

         d_c(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which yields the above formula with :math:`v=1`.
    However, for Poincare ball we can always compute the constant speed :math:`v` from the points
    that particular path connects:

    .. math::

        v = d_c(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_c(x, y)


    Parameters
    ----------
    t : float|tensor
        travelling time
    x : tensor
        starting point on poincare ball
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        point on the Poincare ball
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t).type_as(x)
    return _geodesic(t, x, y, c)


@torch.jit.script
def _geodesic(t, x, y, c):  # pragma: no cover
    # this is not very numerically unstable
    v = _mobius_add(-x, y, c)
    tv = _mobius_scalar_mul(t, v, c)
    gamma_t = _mobius_add(x, tv, c)
    return gamma_t


def expmap(x, u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model. This is tightly related with :func:`geodesic`.
    Intuitively Exponential map is a smooth constant travelling from starting point :math:`x` with speed :math:`u`.

    A bit more formally this is travelling along curve :math:`\gamma_{x, u}(t)` such that

    .. math::

        \gamma_{x, u}(0) = x\\
        \dot\gamma_{x, u}(0) = u\\
        \|\dot\gamma_{x, u}(t)\|_{\gamma_{x, u}(t)} = \|u\|_x

    The existence of this curve relies on uniqueness of differential equation solution, that is local.
    For the Poincare ball model the solution is well defined globally and we have.

    .. math::

        \operatorname{Exp}^c_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_c \tanh(\sqrt{c}/2 \|u\|_x) \frac{u}{\sqrt{c}\|u\|_2}

    Parameters
    ----------
    x : tensor
        starting point on poincare ball
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _expmap(x, u, c)


@torch.jit.script
def _expmap(x, u, c):  # pragma: no cover
    u += 1e-15
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True)
    second_term = (
        tanh(sqrt_c / 2 * _lambda_x(x, c, keepdim=True) * u_norm)
        * u
        / (sqrt_c * u_norm)
    )
    gamma_1 = _mobius_add(x, second_term, c)
    return gamma_1


def expmap0(u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model from :math:`0`.

    .. math::

        \operatorname{Exp}^c_0(u) = \tanh(\sqrt{c}/2 \|u\|_2) \frac{u}{\sqrt{c}\|u\|_2}

    Parameters
    ----------
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


@torch.jit.script
def _expmap0(u, c):  # pragma: no cover
    u = u + 1e-15
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def geodesic_unit(t, x, u, *, c=1.0):
    r"""
    Unit speed geodesic starting from :math:`x` with direction :math:`u/\|u\|_x`

    .. math::

        \gamma_{x,u}(t) = x\oplus_c \tanh(t\sqrt{c}/2) \frac{u}{\sqrt{c}\|u\|_2}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        direction
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _geodesic_unit(t, x, u, c)


@torch.jit.script
def _geodesic_unit(t, x, u, c):  # pragma: no cover
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True)
    second_term = tanh(sqrt_c / 2 * t) * u / (sqrt_c * u_norm)
    gamma_1 = _mobius_add(x, second_term, c)
    return gamma_1


def logmap(x, y, *, c=1.0):
    r"""
    Logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \operatorname{Log}^c_x(y) = \frac{2}{\sqrt{c}\lambda_x^c} \tanh^{-1}(
            \sqrt{c} \|(-x)\oplus_c y\|_2
        ) * \frac{(-x)\oplus_c y}{\|(-x)\oplus_c y\|_2}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on poincare ball
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _logmap(x, y, c)


@torch.jit.script
def _logmap(x, y, c):  # pragma: no cover
    sub = _mobius_add(-x, y, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True)
    lam = _lambda_x(x, c, keepdim=True)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def logmap0(y, *, c=1.0):
    r"""
    Logarithmic map for :math:`y` from :math:`0` on the manifold.


    .. math::

        \operatorname{Log}^c_0(y) = \tanh^{-1}(\sqrt{c}\|y\|_2) \frac{y}{\|y\|_2}

    The result is such that

    .. math::

        y = \operatorname{Exp}^c_0(\operatorname{Log}^c_0(y))

    Parameters
    ----------
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(y)
    return _logmap0(y, c)


@torch.jit.script
def _logmap0(y, c):  # pragma: no cover
    sqrt_c = c ** 0.5
    y = y + 1e-15
    y_norm = y.norm(dim=-1, p=2, keepdim=True)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def mobius_matvec(m, x, *, c=1.0):
    r"""
    Generalization for matrix-vector multiplication to hyperbolic space defined as

    .. math::

        M \otimes_c x = (1/\sqrt{c}) \tanh\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}


    Parameters
    ----------
    m : tensor
        matrix for multiplication
    x : tensor
        point on poincare ball
    c : float|tensor
        negative ball curvature

    Returns
    -------
    tensor
        Mobius matvec result
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _mobius_matvec(m, x, c)


@torch.jit.script
def _mobius_matvec(m, x, c):  # pragma: no cover
    x = x + 1e-15
    x_norm = x.norm(dim=-1, keepdim=True, p=2)
    sqrt_c = c ** 0.5
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return _project(res, c)


def mobius_pointwise_mul(w, x, *, c=1.0):
    r"""
    Generalization for pointwise multiplication to hyperbolic space defined as

    .. math::

        \operatorname{diag}(w) \otimes_c x = (1/\sqrt{c}) \tanh\left(
            \frac{\|\operatorname{diag}(w)x\|_2}{x}\tanh^{-1}(\sqrt{c}\|x\|_2)
        \right)\frac{\|\operatorname{diag}(w)x\|_2}{\|x\|_2}


    Parameters
    ----------
    w : tensor
        weights for multiplication
    x : tensor
        point on poincare ball
    c : float|tensor
        negative ball curvature

    Returns
    -------
    tensor
        Mobius pointwise mul result
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _mobius_pointwise_mul(w, x, c)


@torch.jit.script
def _mobius_pointwise_mul(w, x, c):  # pragma: no cover
    x = x + 1e-15
    x_norm = x.norm(dim=-1, keepdim=True, p=2)
    sqrt_c = c ** 0.5
    wx = x * w
    wx_norm = wx.norm(dim=-1, keepdim=True, p=2)
    res_c = tanh(wx_norm / x_norm * artanh(sqrt_c * x_norm)) * wx / (wx_norm * sqrt_c)
    cond = (wx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return _project(res, c)


def mobius_fn_apply_chain(x, *fns, c=1.0):
    r"""
    Generalization for functions in hyperbolic space.
    First, hyperbolic vector is mapped to a Euclidean space via
    :math:`\operatorname{Log}^c_0` and nonlinear function is applied in this tangent space.
    The resulting vector is then mapped back with :math:`\operatorname{Exp}^c_0`

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
        point on poincare ball
    fns : callable[]
        functions to apply
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        Apply chain result
    """
    if not fns:
        return x
    else:
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c).type_as(x)
        ex = _logmap0(x, c)
        for fn in fns:
            ex = fn(ex)
        y = _expmap0(ex, c)
        return y


def mobius_fn_apply(fn, x, *args, c=1.0, **kwargs):
    r"""
    Generalization for functions in hyperbolic space.
    First, hyperbolic vector is mapped to a Euclidean space via
    :math:`\operatorname{Log}^c_0` and nonlinear function is applied in this tangent space.
    The resulting vector is then mapped back with :math:`\operatorname{Exp}^c_0`

    .. math::

        f^{\otimes_c}(x) = \operatorname{Exp}^c_0(f(\operatorname{Log}^c_0(y)))

    Parameters
    ----------
    x : tensor
        point on poincare ball
    fn : callable
        function to apply
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        Result of function in hyperbolic space
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    ex = _logmap0(x, c)
    ex = fn(ex, *args, **kwargs)
    y = _expmap0(ex, c)
    return y


def mobiusify(fn):
    r"""
    Wraps a function so that is works in hyperbolic space. New function will accept additional argument ``c``

    Parameters
    ----------
    fn : callable
        function in Euclidean space, only its first argument is treated as hyperbolic

    Returns
    -------
    callable
        function working in hyperbolic space
    """

    @functools.wraps(fn)
    def mobius_fn(x, *args, c=1.0, **kwargs):
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c).type_as(x)
        ex = _logmap0(x, c)
        ex = fn(ex, *args, **kwargs)
        y = _expmap0(ex, c)
        return y

    return mobius_fn


def dist2plane(x, a, p, *, c=1.0, keepdim=False, signed=False):
    r"""
    Distance from :math:`x` to a hyperbolic hyperplane in Poincare ball
    that is orthogonal to :math:`a` and contains :math:`p`.

    .. plot:: plots/extended/poincare/distance2plane.py

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
    Let's consider now :math:`p\in \mathbb{D}_c^n` and :math:`a\in T_p\mathbb{D}_c^n\backslash \{\mathbf{0}\}`.

    Slightly deviating from traditional notation let's write :math:`\{a\}_p^\perp`
    highlighting the tight relationship of :math:`a\in T_p\mathbb{D}_c^n\backslash \{\mathbf{0}\}`
    with :math:`p \in \mathbb{D}_c^n`. We then define

    .. math::
        \{a\}_p^\perp := \left\{
            z\in T_p\mathbb{D}_c^n \;:\; \langle z, a\rangle_p = 0
        \right\}

    Recalling that a tangent vector :math:`z` for point :math:`p` yields :math:`x = \operatorname{Exp}^c_p(z)`
    we rewrite the above equation as

    .. math::
        \{a\}_p^\perp := \left\{
            x\in \mathbb{D}_c^n \;:\; \langle \operatorname{Log}_p^c(x), a\rangle_p = 0
        \right\}

    This formulation is something more pleasant to work with.
    Putting all together

    .. math::

        \tilde{H}_{a, p}^c = p + \{a\}^\perp_p\\
        = \left\{
            x \in \mathbb{D}_c^n\;:\;\langle\operatorname{Log}^c_p(x), a\rangle_p = 0
        \right\} \\
        = \left\{
            x \in \mathbb{D}_c^n\;:\;\langle -p \oplus_c x, a\rangle = 0
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
        point on poincare ball
    a : tensor
        point on poincare ball, that is orthogonal to hyperplane
    p : tensor
        point on poincare ball lying on the hyperplane
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    signed : bool
        return signed distance

    Returns
    -------
    tensor
        distance to the hyperplane
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _dist2plane(x, a, p, c, keepdim=keepdim, signed=signed)


@torch.jit.script
def _dist2plane(
    x, a, p, c, keepdim: bool = False, signed: bool = False
):  # pragma: no cover
    sqrt_c = c ** 0.5
    diff = _mobius_add(-p, x, c)
    diff_norm2 = diff.pow(2).sum(dim=-1, keepdim=keepdim)
    sc_diff_a = (diff * a).sum(dim=-1, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=-1, keepdim=keepdim, p=2)
    num = 2 * sqrt_c * sc_diff_a
    denom = (1 - c * diff_norm2) * a_norm
    return arsinh(num / denom) / sqrt_c


def gyration(a, b, u, *, c=1.0):
    r"""
    Gyration is a special operation in hyperbolic geometry.
    Addition operation :math:`\oplus` is not associative (as mentioned in :func:`mobius_add`),
    but gyroassociative which means

    .. math::

        a \oplus (b \oplus c) = (a\oplus b) \oplus \operatorname{gyr}[a, b]c,

    where

    .. math::

        \operatorname{gyr}[a, b]c = \ominus (a \oplus b) \oplus (a \oplus (b \oplus c))

    Parameters
    ----------
    a : tensor
        first point on poincare ball
    b : tensor
        second point on poincare ball
    u : tensor
        vector field for operation
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        the result of automorphism
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(a)
    return _gyration(a, b, u, c)


@torch.jit.script
def _gyration(a, b, u, c):  # pragma: no cover
    mapb = -_mobius_add(a, b, c)
    bpu = _mobius_add(b, u, c)
    apbpu = _mobius_add(a, bpu, c)
    return _mobius_add(mapb, apbpu, c)


def parallel_transport(x, y, v, *, c=1.0):
    r"""
    Parallel transport is essential for adaptive algorithms in Riemannian manifolds.
    For Hyperbolic spaces parallel transport is expressed via gyration.

    .. plot:: plots/extended/poincare/gyrovector_parallel_transport.py

    To recover parallel transport we first need to study isomorphism between gyrovectors and vectors.
    The reason is that originally, parallel transport is well defined for gyrovectors as

    .. math::

        P_{x\to y}(z) = \operatorname{gyr}[y, -x]z,

    where :math:`x,\:y,\:z \in \mathbb{D}_c^n` and
    :math:`\operatorname{gyr}[a, b]c = \ominus (a \oplus b) \oplus (a \oplus (b \oplus c))`

    But we want to obtain parallel transport for vectors, not for gyrovectors.
    The blessing is isomorphism mentioned above. This mapping is given by

    .. math::

        U^c_p \: : \: T_p\mathbb{D}_c^n \to \mathbb{G} = v \mapsto \lambda^c_p v


    Finally, having points :math:`x,\:y \in \mathbb{D}_c^n` and a tangent vector :math:`u\in T_x\mathbb{D}_c^n` we obtain

    .. math::

        P^c_{x\to y}(v) = (U^c_y)^{-1}\left(\operatorname{gyr}[y, -x] U^c_x(v)\right)\\
        = \operatorname{gyr}[y, -x] v \lambda^c_x / \lambda^c_y

    .. plot:: plots/extended/poincare/parallel_transport.py


    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        transported vector
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _parallel_transport(x, y, v, c)


@torch.jit.script
def _parallel_transport(x, y, u, c):  # pragma: no cover
    return (
        _gyration(y, -x, u, c)
        * _lambda_x(x, c, keepdim=True)
        / _lambda_x(y, c, keepdim=True)
    )


def parallel_transport0(y, v, *, c=1.0):
    r"""
    Special case parallel transport with starting point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    y : tensor
        target point
    v : tensor
        vector to be transported
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(y)
    return _parallel_transport0(y, v, c)


@torch.jit.script
def _parallel_transport0(y, v, c):  # pragma: no cover
    return v * (1 - c * y.pow(2).sum(-1, keepdim=True))


def egrad2rgrad(x, grad, *, c=1.0):
    r"""
    Translate Euclidean gradient to Riemannian gradient on tangent space of :math:`x`

    .. math::

        \nabla_x = \nabla^E_x / (\lambda_x^c)^2

    Parameters
    ----------
    x : tensor
        point on the poincare ball
    grad : tensor
        euclidean gradient for :math:`x`
    c : float|tensor
        ball negative curvature

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in T_x\mathbb{D}_c^n`
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _egrad2rgrad(x, grad, c)


@torch.jit.script
def _egrad2rgrad(x, grad, c):  # pragma: no cover
    return grad / _lambda_x(x, c, keepdim=True) ** 2
