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
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(MIN_NORM).log_().to(x.dtype)

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


def egrad2rgrad(x, grad, *, dim=-1):
    """
        docs:
    """
    return _egrad2rgrad(x, grad, dim=dim)


def _egrad2rgrad(x, grad, dim: int = -1):
    grad.narrow(dim, 0, 1).mul_(-1.0)
    grad.addcmul_(_ldot(x, grad, dim=dim, keepdim=True).expand_as(x), x)
    return grad


def project(x, *, dim=-1, eps=None):
    """
        docs
    """
    return _project(x, dim, eps)


def _project(x, dim: int = -1, eps: float = None):
    if x.dim() != 2:
        raise RuntimeError("dimension of input should be atleast 2")
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


def expmap(x, u, *, dim=-1):
    """
        docs
    """
    return _expmap(x, u, dim=dim)


def _expmap(x, u, c, dim: int = -1):
    denom = th.sqrt(ldot(u, u, keepdim=True))
    p = (th.cosh(t) * p) + (th.sinh(t) * p / denom)
    return p


def logmap(x, y, *, dim=-1):
    """
        docs
    """
    return _logmap(x, y, dim=dim)


def _logmap(x, y, *, dim: int = -1):
    xy = ldot(x, y, keepdim=True)
    denom = th.sqrt(xy * xy - 1)
    v = (Acosh.apply(-xy, self.eps) / tmp) * th.addcmul(y, xy, x)
    return v


def egrad2rgrad(x, grad, *, dim=-1):
    """
        docs
    """
    return _egrad2rgrad(x, grad, dim=dim)


def _egrad2rgrad(x, grad, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad.addcmul_(self.ldot(x, grad, keepdim=True).expand_as(x), x)
    return grad
