import functools
import torch.jit
import torch as th

MIN_NORM = 1e-15


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(th.clamp_min(x.pow(2) - 1.0, MIN_NORM))
        ctx.save_for_backward(z)
        ctx.eps = eps
        asd = th.log(x + z)
        return asd

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


def _arcosh(x, eps=1e-6):
    return Arcosh.apply(x, eps)


def dist(x, y, *, k=1.0, keepdim=False, dim=-1, eps=1e-6):
    r"""
    Compute geodesic distance on the Hyperboloid

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
    if isinstance(k, float):
        k = th.Tensor([k])
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim, eps=eps)


def _dist(x, y, k, keepdim: bool = False, dim: int = -1, eps=1e-6):
    d = -inner(x, y, dim=dim, keepdim=keepdim)
    return th.sqrt(k) * _arcosh(d / k, eps)


def dist0(x, *, k=1.0, keepdim=False, dim=-1, eps=1e-6):
    r"""
    Compute geodesic distance on the Hyperboloid to zero.

    Parameters
    ----------
    x : tensor
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
        geodesic distance between :math:`x` and :math:`0`
    """
    if isinstance(k, float):
        k = th.Tensor([k])
    return _dist0(x, k, keepdim=keepdim, dim=dim, eps=eps)


def _dist0(x, k, keepdim: bool = False, dim: int = -1, eps=1e-6):
    zp = th.ones_like(x)
    d = zp.size(dim) - 1
    zp = th.cat(
        (zp.narrow(dim, 0, 1) * th.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )
    d = -inner(x, zp, dim=dim, keepdim=keepdim)
    return th.sqrt(k) * _arcosh(d / k, eps)


def project(x, *, k=1.0, dim=-1):
    r"""
    Projection on the Hyperboloid

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{1+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: float|tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


def _project(x, k, dim: int = -1):
    dn = x.size(dim) - 1
    left_ = th.sqrt(k + th.norm(x.narrow(dim, 1, dn), dim=dim) ** 2).unsqueeze(dim)
    right_ = x.narrow(dim, 1, dn)
    proj = torch.cat((left_, right_), dim=dim)
    return proj


def project_polar(x, *, k=1.0, dim=-1):
    r"""
    Projection on the Hyperboloid from polar coordinates

    ... math::
        \pi((\mathbf{d}, r))=(\sinh (r) \mathbf{d}, \cosh (r))

    Parameters
    ----------
    x: tensor
        point in Rn
    k: float|tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    if isinstance(k, float):
        k = th.Tensor([k])
    return _project_polar(x, k=k, dim=dim)


def _project_polar(x, k, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = th.cat(
        (th.cosh(r / th.sqrt(k)), th.sqr(k) * th.sinh(r / th.sqrt(k)) * d), dim=dim
    )
    return res


def project_u(x, v, *, k=1.0, dim=-1):
    r"""
    Projection of the vector on the tangent space of the Hyperboloid

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x}

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    v: tensor
        vector in Rn
    k: float|tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_u(x, v, k=k, dim=dim)


def _project_u(x, v, k, dim=-1):
    return v.addcmul(inner(x, v, dim=dim, keepdim=True).expand_as(x), x / k)


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
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Hyperboloid

    .. math::

        \|\mathbf{v}\|_{\mathcal{L}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle_{\mathcal{L}}}

    Parameters
    ----------
    u : tensor
        tangent vector on Hyperboloid
    k : float|tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    """
    return _norm(u, keepdim=keepdim, dim=dim)


def _norm(u, keepdim: bool = False, dim: int = -1):
    return th.sqrt(inner(u, u, keepdim=keepdim))


def expmap(x, u, *, k=1.0, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid

    .. math::

        \exp _{\mathbf{x}}^{K}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{K}}\right) \mathbf{x}+\sqrt{K} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{K}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
    u : tensor
        unit speed vector on Hyperboloid
    c : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    if isinstance(k, float):
        k = th.Tensor([k])
    return _expmap(x, u, k=k, dim=dim)


def _expmap(x, u, k, dim: int = -1):
    nomin = norm(u, keepdim=True, dim=dim)
    p = (
        th.cosh(nomin / th.sqrt(k)) * x
        + th.sqrt(k) * th.sinh(nomin / th.sqrt(k)) * u / nomin
    )
    return p


def expmap0(u, *, k=1.0, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    if isinstance(k, float):
        k = th.Tensor([k])
    return _expmap0(u, k, dim=dim)


def _expmap0(u, k, dim: int = -1):
    zp = th.ones_like(u)
    d = zp.size(dim) - 1
    zp = th.cat(
        (zp.narrow(dim, 0, 1) * th.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )
    nomin = norm(u, keepdim=True, dim=dim)
    p = (
        th.cosh(nomin / th.sqrt(k)) * zp
        + th.sqrt(k) * th.sinh(nomin / th.sqrt(k)) * u / nomin
    )
    return p


def logmap(x, y, *, k=1.0, dim=-1):
    r"""
    Compute logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \log _{\mathbf{x}}^{K}(\mathbf{y})=d_{\mathcal{L}}^{K}(\mathbf{x}, \mathbf{y})
            \frac{\mathbf{y}+\frac{1}{K}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}}{\left\|
            \mathbf{y}+\frac{1}{K}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}\right\|_{\mathcal{L}}}

    The result of Logarithmic map is a vector such that

    .. math::

        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
    y : tensor
        target point on Hyperboloid
    k : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k=k, dim=dim)


def _logmap(x, y, k, dim: int = -1):
    dist_ = dist(x, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * inner(x, y, keepdim=True) * x
    denom = norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0(y, *, k=1.0, dim=-1):
    r"""
    Compute logarithmic map for :math:`y` from :math:`0` on the manifold.

    Parameters
    ----------
    y : tensor
        target point on Hyperboloid
    k : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k=k, dim=dim)


def _logmap0(y, k, dim: int = -1):
    zp = th.ones_like(y)
    d = zp.size(dim) - 1
    zp = th.cat(
        (zp.narrow(dim, 0, 1) * th.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )

    dist_ = dist(zp, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * inner(zp, y, keepdim=True) * zp
    denom = norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def egrad2rgrad(x, grad, *, k=1.0, dim=-1):
    r"""
    Translate Euclidean gradient to Riemannian gradient on tangent space of :math:`x`.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, k}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \frac{\mathbf{x}}{k}

    Parameters
    ----------
    x : tensor
        point on the Hyperboloid
    grad : tensor
        Euclidean gradient for :math:`x`
    k : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in `
    """
    return _egrad2rgrad(x, grad, k=k, dim=dim)


def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad = grad.addcmul(inner(x, grad, dim=dim, keepdim=True).expand_as(x), x / k)
    return grad


def parallel_transport(x, y, v, *, k=1.0, dim=-1):
    r"""
    Perform parallel transport on the Hyperboloid

    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    c : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k=k, dim=dim)


def _parallel_transport(x, y, v, k, dim: int = -1):
    nom = inner(logmap(x, y, k=k, dim=dim), v, keepdim=True)
    denom = dist(x, y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (logmap(x, y, k=k, dim=dim) + logmap(y, x, k=k, dim=dim))
    return p


def parallel_transport0(y, v, *, k=1.0, dim=-1):
    r"""
    Perform parallel transport from zero point.

    Parameters
    ----------
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    c : float|tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport0(y, v, k=k, dim=dim)


def _parallel_transport0(y, v, k, dim: int = -1):
    zp = th.ones_like(y)
    d = zp.size(dim) - 1
    zp = th.cat(
        (zp.narrow(dim, 0, 1) * th.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )

    nom = inner(logmap(zp, y, k=k, dim=dim), v, keepdim=True)
    denom = dist(zp, y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (logmap(zp, y, k=k, dim=dim) + logmap(y, zp, k=k, dim=dim))
    return p


def geodesic_unit(t, x, u, k=1.0):
    r"""
    Compute unit speed geodesic at time :math:`t` starting from :math:`x` with direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{\mathbf{x} \rightarrow \mathbf{u}}^{K}(t)=\cosh \left(\frac{t}{\sqrt{K}}\right) \mathbf{x}+\sqrt{K} \sinh \left(\frac{t}{\sqrt{K}}\right) \mathbf{u}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        unit direction vector
    k : float|tensor
        manifold negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    if isinstance(k, float):
        k = th.Tensor([k])
    return _geodesic_unit(t, x, u, k=k)


def _geodesic_unit(t, x, u, k):
    return th.cosh(t / th.sqrt(k)) * x + th.sqrt(k) * th.sinh(t / th.sqrt(k)) * u


def lorentz_to_poincare(x, k=1.0, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+1}

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disk
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + k)


def poincare_to_lorentz(x, k=1.0, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disk to Hyperboloid

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Hyperboloid
    """
    if isinstance(k, float):
        k = th.Tensor([k])
    x_norm_square = th.sum(x * x, dim=dim, keepdim=True)
    res = (
        th.sqrt(k)
        * th.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res
