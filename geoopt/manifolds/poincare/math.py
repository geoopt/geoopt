import torch
import torch.jit


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
    norm = x.norm(-1, keepdim=True)
    maxnorm = (1 - 1e-5) / (c ** 0.5 + 1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x, *, c):
    r"""
    Compute the conformal factor for a point on the ball

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature

    Returns
    -------
    scalar
    """
    return 2 / (1 - c * x.pow(2).sum(-1))


def inner(x, u, v, *, c):
    r"""
    Compute inner product for two vectors on the tangent space w.r.t Riemannian metric on the Poincare ball

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
    return lambda_x(x, c=c) ** 2 * (u * v).sum(-1)


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

    * zero vector vase

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
    y = y + 1e-15  # add small epsilon for stability
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
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
    return mobius_add(x, -y, c=c)
