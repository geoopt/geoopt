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


def project(x, *, c):
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


def lambda_x(x, *, c):
    r"""
    Compute the conformal factor :math:`\lambda_x` for a point on the ball

    .. math::

        \lambda_x = \frac{1}{1 - c \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    scalar
        conformal factor
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _lambda_x(x, c)


@torch.jit.script
def _lambda_x(x, c):  # pragma: no cover
    return 2 / (1 - c * x.pow(2).sum(-1))


def inner(x, u, v, *, c):
    r"""
    Compute inner product for two vectors on the tangent space w.r.t Riemannian metric on the Poincare ball

    .. math::

        \langle u, v\rangle_x = \lambda_x^2 \langle u, v \rangle

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

    Returns
    -------
    scalar
        inner product
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _inner(x, u, v, c)


@torch.jit.script
def _inner(x, u, v, c):  # pragma: no cover
    return _lambda_x(x, c) ** 2 * (u * v).sum(-1)


def mobius_add(x, y, *, c):
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


def mobius_sub(x, y, *, c):
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


def mobius_scalar_mul(r, x, *, c):
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
def _mobius_scalar_mul(r, x, c):
    x = x + 1e-15
    x_norm = x.norm(dim=-1, keepdim=True, p=2)
    sqrt_c = c ** 0.5
    res_c = tanh(r * artanh(sqrt_c * x_norm)) * x / (x_norm * sqrt_c)
    return _project(res_c, c)


def dist(x, y, *, c):
    r"""
    Distance on the Poincare ball

    .. math::

        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}\|(\sqrt{c}(-c)\oplus_c y\|_2)

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
    scalar
        geodesic distance between :math:`x` and :math:`y`
    """
    if not isinstance(c, torch.Tensor):
        c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c)


@torch.jit.script
def _dist(x, y, c):
    sqrt_c = c ** 0.5
    dist_c = 2 / sqrt_c * artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2))
    return dist_c
