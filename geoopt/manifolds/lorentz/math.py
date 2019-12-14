import functools
import torch.jit
import torch as th

MIN_NORM = 1e-15
_eps = 1e-10


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(th.clamp(x * x - 1 + eps, _eps))
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt(1 + z.pow(2))).clamp_min(MIN_NORM).log().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def arsinh(x):
    return Arsinh.apply(x)


def arcosh(x):
    return Arcosh.apply(x)


def dist(x, y, *, k=1.0, keepdim=False, dim=-1, eps=1e-3):
    r"""
    Compute geodesic distance on the Poincare ball.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}} / k\right)

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : float|tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension
    eps : float
        stability parameter for arcosh

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim, eps=eps)


def _dist(x, y, k, keepdim: bool = False, dim: int = -1, eps=1e-3):
    d = -inner(x, y, dim=dim, keepdim=keepdim)
    return th.sqrt(k) * arcosh(d / k, eps)


def project(x, *, k=1.0, dim=-1):
    r"""
    Projection on the hyperboloid

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    k: float|tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, dim)


def _project(x, dim: int = -1):
    dn = x.size(dim) - 1
    left_ = th.sqrt(1.0 + th.norm(x.narrow(dim, 0, dn), dim=dim) ** 2).unsqueeze(dim)
    right_ = x.narrow(dim, 0, dn)
    proj = torch.cat((left_, right_), dim=dim)
    return proj


def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product

    .. math::
        \langle\mathbf{u}, \mathbf{v}\rangle_{\mathcal{L}}:=-u_{0} v_{0}+u_{1} v_{1}+\ldots+u_{d} v_{d}

    Parameters
    ----------
    u : tensor
        vector in ambient space
    v : tensor
        vector in ambient space
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(u, v, keepdim=keepdim, dim=dim)


def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    uv = th.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim)
    return th.sum(uv, dim=dim, keepdim=keepdim)


def norm(u, *, keepdim=False, dim=-1):
    """
        TODO: docs
    """
    return _norm(u, keepdim=keepdim, dim=dim)


def _norm(u, keepdim: bool = False, dim: int = -1):
    return th.sqrt(th.inner(u, u))


def normalize_tan(x_all, v_all, dim=-1):
    """
        docs
    """
    d = v_all.size(dim) - 1
    x = x_all.narrow(dim, 1, d)
    xv = th.sum(x * v_all.narrow(dim, 1, d), dim=dim, keepdim=True)
    tmp = 1 + th.sum(th.pow(x_all.narrow(dim, 1, d), 2), dim=dim, keepdim=True)
    tmp = th.sqrt(tmp)
    return th.cat((xv / tmp, v_all.narrow(dim, 1, d)), dim=dim)


def norm(u, *, keepdim=False, dim=-1):
    r"""
        ...
    """
    return _norm(u, keepdim=keepdim, dim=dim)


def _norm(u, keepdim=False, dim=-1):
    return th.sqrt(inner(u, u, keepdim=keepdim, dim=dim))


def expmap(x, u, *, k=1.0, dim=-1):
    """
        docs
    """
    return _expmap(x, u, k=k, dim=dim)


def _expmap(x, u, k, dim: int = -1):
    u = normalize_tan(x, u, dim=dim)
    denom = norm(u, keepdim=True, dim=dim)
    p = (th.cosh(denom / th.sqrt(k)) * x) + th.sqrt(k) * (th.sinh(denom / th.sqrt(k)) * u / denom)
    return p


def logmap(x, y, *, k=1.0, dim=-1, eps=1e-3):
    """
        docs
    """
    return _logmap(x, y, k=k, dim=dim, eps=eps)


def _logmap(x, y, k, dim: int = -1, eps=1e-3):
    dist_ = dist(x, y, k, dim=dim, keepdim=True)
    nomin = y + 1. / k * inner(x, y) * x
    denom = norm(nomin)
    return dist_ * nomin / denom


def egrad2rgrad(x, grad, *, dim=-1):
    """
        docs
    """
    return _egrad2rgrad(x, grad, dim=dim)


def _egrad2rgrad(x, grad, dim: int = -1):
    grad = grad.narrow(dim, 0, 1).mul(-1.0)
    grad = grad.addcmul(inner(x, grad, dim=dim, keepdim=True).expand_as(x), x)
    return grad


def parallel_transport(x, y, v, *, dim=-1):
    """
        docs
    """
    return _parallel_transport(x, y, v, dim=dim)


def _parallel_transport(x, y, v, dim: int = -1):
    nom = inner(logmap(x, y, dim=dim), v)
    denom = dist(x, y, dim=dim) ** 2
    p = v - nom / denom * (logmap(x, y, dim=dim) + logmap(y, x, dim=dim))
    return p


def geodesic_unit(t, x, u, *, dim=-1):
    """
        docs
    """
    return _geodesic_unit(t, x, u, dim=dim)


def _geodesic_unit(t, x, u, dim: int = -1):
    return th.cosh(t / th.sqrt(k)) * x + th.sqrt(k) * th.sinh(t / th.sqrt(k)) * u


###


def lorentz_to_poincare(x, dim=-1):
    """
        docs
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + 1.0)


def poincare_to_lorentz(x, dim=-1, eps=1e-3):
    """
        docs
    """
    x_norm_square = th.sum(x * x, dim=dim, keepdim=True)
    return th.cat((1.0 + x_norm_square, 2 * x), dim=dim) / (1 - x_norm_square + eps)
