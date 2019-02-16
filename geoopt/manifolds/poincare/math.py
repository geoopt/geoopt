import functools
import torch
import torch.jit


@torch.jit.script
def tanh(x):  # pragma: no cover
    return x.clamp(-15, 15).tanh()


# noinspection PyTypeChecker,PyUnresolvedReferences
@torch.jit.script
def artanh(x):  # pragma: no cover
    res = (0.5 * (torch.log(1 + x) - torch.log(1 - x))).clamp(-1 + 1e-5, 1 - 1e-5)
    return res


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
    maxnorm = (1 - 1e-5) / (c ** 0.5)
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
    return _lambda_x(x, c) ** 2 * (u * v).sum(-1, keepdim=keepdim)


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
    return _lambda_x(x, c) * u.norm(dim=-1, keepdim=keepdim, p=2)


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
    return num / denom


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

        \operatorname{Exp}_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_c \tanh(\sqrt{c}/2 \|u\|_x) \frac{u}{\sqrt{c}\|u\|}

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

        \operatorname{Exp}_0(u) = \tanh(\sqrt{c}/2 \|u\|_2) \frac{u}{\sqrt{c}\|u\|}

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

        \gamma_{x,u}(t) = x\oplus_c \tanh(t\sqrt{c}/2) \frac{u}{\sqrt{c}\|u\|}

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

        \operatorname{Log}_x(y) = \frac{2}{\sqrt{c}\lambda_x^c} \tanh^{-1}(
            \sqrt{c} \|(-x)\oplus_c y\|_2
        ) * \frac{(-x)\oplus_c y}{\|(-x)\oplus_c y\|_2}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}_x(\operatorname{Log}_x(y))


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

        \operatorname{Log}_0(y) = \tanh^{-1}(\sqrt{c}\|y\|_2) \frac{y}{\|y\|_2}

    The result is such that

    .. math::

        y = \operatorname{Exp}_0(\operatorname{Log}_0(y))

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

        M \otimes_c x = (1/\sqrt{c}) \tanh\left(\frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)\right)\frac{Mx}{\|Mx\|_2}


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
    :math:`\operatorname{Log}_0` and nonlinear function is applied in this tangent space.
    The resulting vector is then mapped back with :math:`\operatorname{Exp}_0`

    .. math::

        f^{\otimes_c}(x) = \operatorname{Exp}_0(f(\operatorname{Log}_0(y)))

    The definition of mobius function application allows chaining as

    .. math::

        y = \operatorname{Exp}_0(\operatorname{Log}_0(y))

    Resulting in

    .. math::

        (f \circ g)^{\otimes_c}(x) = \operatorname{Exp}_0((f \circ g) (\operatorname{Log}_0(y)))

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
    :math:`\operatorname{Log}_0` and nonlinear function is applied in this tangent space.
    The resulting vector is then mapped back with :math:`\operatorname{Exp}_0`

    .. math::

        f^{\otimes_c}(x) = \operatorname{Exp}_0(f(\operatorname{Log}_0(y)))

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
