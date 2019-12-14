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


def ldot(u, v, *, dim: int = -1, keepdim=False):
    """
        Lorentzian Scalar Product
    """
    return _ldot(u, v, dim=dim, keepdim=keepdim)


def _ldot(u, v, dim: int = -1, keepdim=False):
    d = u.size(dim) - 1
    uv = u * v
    uv = th.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim)
    return th.sum(uv, dim=dim, keepdim=keepdim)


def dist(x, y, *, keepdim=False, dim=-1, eps=1e-3):
    """
        docs: ...
    """
    return _dist(x, y, keepdim=keepdim, dim=dim, eps=eps)


def _dist(x, y, keepdim: bool = False, dim: int = -1, eps=1e-3):
    d = -ldot(x, y, dim=dim, keepdim=keepdim)
    return arcosh(d, eps)


def project(x, *, dim=-1, eps=None):
    """
        docs
    """
    return _project(x, dim, eps)


def _project(x, dim: int = -1, eps: float = None):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    d = d / th.norm(d, dim=dim, keepdim=True)
    r = x.narrow(dim, -1, 1)
    proj = torch.cat((torch.cosh(r), torch.sinh(r) * d), dim=dim)
    return proj


def inner(x, u, v, *, keepdim=False, dim=-1):
    """
        docs
    """
    return _inner(x, u, v, keepdim=keepdim, dim=dim)


def _inner(x, u, v, keepdim: bool = False, dim: int = -1):
    return ldot(u, v)


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


def norm(x, u, *, keepdim=False, dim=-1):
    """
        docs
    """
    return _norm(x, u, keepdim=keepdim, dim=dim)


def _norm(x, u, keepdim=False, dim=-1):
    return th.sqrt(ldot(u, u, keepdim=keepdim, dim=dim))


def expmap(x, u, *, dim=-1):
    """
        docs
    """
    return _expmap(x, u, dim=dim)


def _expmap(x, u, dim: int = -1):
    u = normalize_tan(x, u, dim=dim)
    denom = norm(x, u, keepdim=True, dim=dim)
    p = (th.cosh(denom) * x) + (th.sinh(denom) * u / denom)
    return p


def logmap(x, y, *, dim=-1):
    """
        docs
    """
    return _logmap(x, y, dim=dim)


def _logmap(x, y, *, dim: int = -1, eps=1e-3):
    xy = ldot(x, y, keepdim=True, dim=dim)
    denom = th.sqrt(xy * xy - 1.0)
    v = (Arcosh.apply(-xy, eps) / denom) * th.addcmul(y, xy, x)
    return v


def egrad2rgrad(x, grad, *, dim=-1):
    """
        docs
    """
    return _egrad2rgrad(x, grad, dim=dim)


def _egrad2rgrad(x, grad, dim: int = -1):
    grad = grad.narrow(dim, 0, 1).mul(-1.0)
    grad = grad.addcmul(ldot(x, grad, dim=dim, keepdim=True).expand_as(x), x)
    return grad


def parallel_transport(x, y, v, *, dim=-1):
    """

    """
    return _parallel_transport(x, y, v, dim=dim)


def _parallel_transport(x, y, v, dim: int = -1):
    nom = ldot(logmap(x, y, dim=dim), v)
    denom = dist(x, y, dim=dim) ** 2
    p = v - nom / denom * (logmap(x, y, dim=dim) + logmap(y, x, dim=dim))
    return p


def geodesic_unit(t, x, u, *, dim=-1):
    """
        docs
    """
    return _geodesic_unit(t, x, u, dim=dim)


def _geodesic_unit(t, x, u, dim: int = -1):
    return th.cosh(t) * x + th.sinh(t) * u


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
