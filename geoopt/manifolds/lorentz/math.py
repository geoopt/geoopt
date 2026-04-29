from typing import List, Optional

import torch.jit

from ...utils import drop_dims


@torch.jit.script
def arcosh(x: torch.Tensor):
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))
    return torch.log(x + z).to(dtype)


def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product.

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


@torch.jit.script
def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
            dim=dim, keepdim=True
        )


def inner0(v, *, k, keepdim=False, dim=-1):
    r"""
    Minkowski inner product with zero vector.

    Parameters
    ----------
    v : tensor
        vector in ambient space
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner0(v, k=k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1) * torch.sqrt(k)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def pairwise_inner(U, V, *, dim=-1):
    r"""
    Compute pairwise Minkowski inner products between two batches of vectors.

    Parameters
    ----------
    U : tensor
        Batch of vectors in ambient space, shape (..., B1, D)
    V : tensor
        Batch of vectors in ambient space, shape (..., B2, D)
    dim : int
        Reduction dimension

    Returns
    -------
    tensor
        Pairwise inner product matrix, shape (..., B1, B2)
    """
    return _pairwise_inner(U, V, dim=dim)


@torch.jit.script
def _pairwise_inner(U, V, dim: int = -1):
    d = U.size(dim) - 1
    U_time = U.narrow(dim, 0, 1)
    U_space = U.narrow(dim, 1, d)
    V_time = V.narrow(dim, 0, 1)
    V_space = V.narrow(dim, 1, d)
    time_product = -torch.einsum(
        "...i,...j->...ij", U_time.squeeze(dim), V_time.squeeze(dim)
    )
    space_product = torch.einsum("...id,...jd->...ij", U_space, V_space)

    return time_product + space_product


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\frac{\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}}}{k}\right)

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def dist0(x, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero point.

    .. math::

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and zero point
    """
    return _dist0(x, k=k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(x, k=k, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def cdist(X, Y, *, k):
    r"""
    Compute pairwise geodesic distances on the Hyperboloid between two batches of points.

    Parameters
    ----------
    X : tensor
        Batch of points on Hyperboloid, shape (..., B1, D)
    Y : tensor
        Batch of points on Hyperboloid, shape (..., B2, D)
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        Pairwise geodesic distance matrix, shape (..., B1, B2)
    """
    return _cdist(X, Y, k=k)


@torch.jit.script
def _cdist(X, Y, k: torch.Tensor):
    D = -_pairwise_inner(X, Y, dim=-1)
    return torch.sqrt(k) * arcosh(D / k)


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{k+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k=k, dim=dim)


@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    left_ = torch.sqrt(
        k + torch.norm(x.narrow(dim, 1, dn), p=2, dim=dim) ** 2
    ).unsqueeze(dim)
    right_ = x.narrow(dim, 1, dn)
    proj = torch.cat((left_, right_), dim=dim)
    return proj


def project_polar(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid from polar coordinates.

    ... math::
        \pi((\mathbf{d}, r))=(\sqrt{k} \sinh (r/\sqrt{k}) \mathbf{d}, \cosh (r / \sqrt{k}))

    Parameters
    ----------
    x: tensor
        point in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_polar(x, k=k, dim=dim)


@torch.jit.script
def _project_polar(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = torch.cat(
        (
            torch.cosh(r / torch.sqrt(k)),
            torch.sqrt(k) * torch.sinh(r / torch.sqrt(k)) * d,
        ),
        dim=dim,
    )
    return res


def project_u(x, v, *, k, dim=-1):
    r"""
    Projection of the vector on the tangent space.

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x} / k

    Parameters
    ----------
    x: tensor
        point on the Hyperboloid
    v: tensor
        vector in Rn
    k: tensor
        hyperboloid negative curvature
    dim : int
        reduction dimension to compute norm

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project_u(x, v, k=k, dim=dim)


@torch.jit.script
def _project_u(x, v, k: torch.Tensor, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x / k)


def norm(u, *, keepdim=False, dim=-1):
    r"""
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Hyperboloid.

    .. math::

        \|\mathbf{v}\|_{\mathcal{L}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle_{\mathcal{L}}}

    Parameters
    ----------
    u : tensor
        tangent vector on Hyperboloid
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


@torch.jit.script
def _norm(u, keepdim: bool = False, dim: int = -1):
    return torch.sqrt(torch.clamp_min(_inner(u, u, keepdim=keepdim), 1e-8))


def expmap(x, u, *, k, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid.

    .. math::

        \exp _{\mathbf{x}}^{k}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{k}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    u : tensor
        unit speed vector on Hyperboloid
    k: tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    return _expmap(x, u, k=k, dim=dim)


@torch.jit.script
def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    nomin = _norm(u, keepdim=True, dim=dim)
    p = (
        torch.cosh(nomin / torch.sqrt(k)) * x
        + torch.sqrt(k) * torch.sinh(nomin / torch.sqrt(k)) * u / nomin
    )
    return p


def expmap0(u, *, k, dim=-1):
    r"""
    Compute exponential map for Hyperboloid from :math:`0`.

    Parameters
    ----------
    u : tensor
        speed vector on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    return _expmap0(u, k, dim=dim)


@torch.jit.script
def _expmap0(u, k: torch.Tensor, dim: int = -1):
    nomin = _norm(u, keepdim=True, dim=dim)
    l_v = torch.cosh(nomin / torch.sqrt(k)) * torch.sqrt(k)
    r_v = torch.sqrt(k) * torch.sinh(nomin / torch.sqrt(k)) * u / nomin
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p


def logmap(x, y, *, k, dim=-1):
    r"""
    Compute logarithmic map for two points :math:`x` and :math:`y` on the manifold.

    .. math::

        \log _{\mathbf{x}}^{k}(\mathbf{y})=d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})
            \frac{\mathbf{y}+\frac{1}{k}\langle\mathbf{x},
            \mathbf{y}\rangle_{\mathcal{L}} \mathbf{x}}{\left\|
            \mathbf{y}+\frac{1}{k}\langle\mathbf{x},
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
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    return _logmap(x, y, k=k, dim=dim)


@torch.jit.script
def _logmap(x, y, k, dim: int = -1):
    dist_ = _dist(x, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * _inner(x, y, keepdim=True) * x
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0(y, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`y` from :math:`0` on the manifold.

    Parameters
    ----------
    y : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k=k, dim=dim)


@torch.jit.script
def _logmap0(y, k, dim: int = -1):
    dist_ = _dist0(y, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(y, k=k, keepdim=True) * torch.sqrt(k)
    dn = y.size(dim) - 1
    nomin = torch.cat((nomin_ + y.narrow(dim, 0, 1), y.narrow(dim, 1, dn)), dim)
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0back(x, *, k, dim=-1):
    r"""
    Compute logarithmic map for :math:`0` from :math:`x` on the manifold.

    Parameters
    ----------
    x : tensor
        target point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    return _logmap0back(x, k=k, dim=dim)


@torch.jit.script
def _logmap0back(x, k, dim: int = -1):
    dist_ = _dist0(x, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(x, k=k, keepdim=True) * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1) + torch.sqrt(k), nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def egrad2rgrad(x, grad, *, k, dim=-1):
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
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in `
    """
    return _egrad2rgrad(x, grad, k=k, dim=dim)


@torch.jit.script
def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True), x / k)
    return grad


def parallel_transport(x, y, v, *, k, dim=-1):
    r"""
    Perform parallel transport on the Hyperboloid.

    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport(x, y, v, k=k, dim=dim)


@torch.jit.script
def _parallel_transport(x, y, v, k, dim: int = -1):
    lmap = _logmap(x, y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist(x, y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap(y, x, k=k, dim=dim))
    return p


def parallel_transport0(y, v, *, k, dim=-1):
    r"""
    Perform parallel transport from zero point.

    Parameters
    ----------
    y : tensor
        end point
    v : tensor
        tangent vector to be transported
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    """
    return _parallel_transport0(y, v, k=k, dim=dim)


@torch.jit.script
def _parallel_transport0(y, v, k, dim: int = -1):
    lmap = _logmap0(y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0back(y, k=k, dim=dim))
    return p


def parallel_transport0back(x, v, *, k, dim: int = -1):
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
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    """
    return _parallel_transport0back(x, v, k=k, dim=dim)


@torch.jit.script
def _parallel_transport0back(x, v, k, dim: int = -1):
    lmap = _logmap0back(x, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(x, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))
    return p


def gyroadd(x, y, *, k, dim=-1):
    r"""
    Compute gyroaddition on the Lorentz model.

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    y : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        gyroaddition of :math:`x` and :math:`y`
    """
    return _gyroadd(x, y, k=k, dim=dim)


@torch.jit.script
def _gyroadd(x, y, k: torch.Tensor, dim: int = -1):
    d = x.size(dim) - 1
    x_t = x.narrow(dim, 0, 1)
    y_t = y.narrow(dim, 0, 1)
    x_s = x.narrow(dim, 1, d)
    y_s = y.narrow(dim, 1, d)

    sqrt_k = torch.sqrt(k)
    inv_k = 1.0 / k

    a = 1.0 + x_t / sqrt_k
    b = 1.0 + y_t / sqrt_k

    norm_x = (x_s * x_s).sum(dim=dim, keepdim=True)
    norm_y = (y_s * y_s).sum(dim=dim, keepdim=True)
    inner_xy = (x_s * y_s).sum(dim=dim, keepdim=True)

    d_term = (
        a * a * b * b
        + 2.0 * inv_k * a * b * inner_xy
        + inv_k * inv_k * norm_x * norm_y
    )
    n_term = a * a * norm_y + 2.0 * a * b * inner_xy + b * b * norm_x

    denom = d_term - inv_k * n_term
    sign = torch.sign(denom)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom = denom + sign * 1e-15

    time = sqrt_k * (d_term + inv_k * n_term) / denom

    coef_x = a * b * b + 2.0 * inv_k * b * inner_xy + inv_k * a * norm_y
    coef_y = b * (a * a - inv_k * norm_x)
    space = 2.0 * (coef_x * x_s + coef_y * y_s) / denom
    return _project(torch.cat((time, space), dim=dim), k=k, dim=dim)


def gyroinv(x, *, k=None, dim=-1):
    r"""
    Compute the gyroinverse of a Lorentz point.

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    dim : int
        manifold dimension

    Returns
    -------
    tensor
        gyroinverse of :math:`x`
    """
    return _gyroinv(x, dim=dim)


@torch.jit.script
def _gyroinv(x, dim: int = -1):
    d = x.size(dim) - 1
    return torch.cat((x.narrow(dim, 0, 1), -x.narrow(dim, 1, d)), dim=dim)


def gyroscalar(r, x, *, k, dim=-1):
    r"""
    Compute gyro-scalar multiplication on the Lorentz model.

    Parameters
    ----------
    r : tensor
        scalar multiplier
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        manifold dimension

    Returns
    -------
    tensor
        gyro-scalar multiplication of :math:`x` by :math:`r`
    """
    if not torch.is_tensor(r):
        r = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    if r.dim() == x.dim() - 1:
        r = r.unsqueeze(dim)
    return _gyroscalar(r, x, k=k, dim=dim)


@torch.jit.script
def _gyroscalar(r, x, k: torch.Tensor, dim: int = -1):
    d = x.size(dim) - 1
    x_t = x.narrow(dim, 0, 1)
    x_s = x.narrow(dim, 1, d)

    sqrt_k = torch.sqrt(k)
    norm_s = torch.norm(x_s, p=2, dim=dim, keepdim=True).clamp_min(1e-15)
    theta = arcosh((x_t / sqrt_k).clamp_min(1.0))
    rtheta = r * theta

    time = sqrt_k * torch.cosh(rtheta)
    space = sqrt_k * torch.sinh(rtheta) * x_s / norm_s
    return _project(torch.cat((time, space), dim=dim), k=k, dim=dim)


def lorentz_centroid(
    x,
    weights: Optional[torch.Tensor] = None,
    *,
    k,
    reducedim: Optional[List[int]] = None,
    dim=-1,
    keepdim=False,
    eps=1e-8,
):
    r"""
    Compute the weighted Lorentz centroid.

    Parameters
    ----------
    x : tensor
        points on Hyperboloid
    weights : tensor
        optional non-negative weights, broadcastable to ``x`` without the
        manifold dimension
    k : tensor
        manifold negative curvature
    reducedim : list[int]
        dimensions to reduce. By default all non-manifold dimensions are reduced
    dim : int
        manifold dimension
    keepdim : bool
        retain reduced dimensions
    eps : float
        numerical stability constant

    Returns
    -------
    tensor
        Lorentz centroid
    """
    reducedim = _default_reducedim(x, reducedim, dim)
    avg = _weighted_average(x, weights, reducedim, dim)
    denom = torch.sqrt(torch.clamp_min(-_inner(avg, avg, keepdim=True, dim=dim), eps))
    centroid = torch.sqrt(k) * avg / denom
    centroid = _project(centroid, k=k, dim=dim)
    if not keepdim:
        centroid = drop_dims(centroid, reducedim)
    return centroid


def lorentz_dispersion(
    x,
    mean,
    *,
    k,
    reducedim: Optional[List[int]] = None,
    dim=-1,
    keepdim=False,
):
    r"""
    Compute mean squared tangent dispersion around a Lorentz mean.

    Parameters
    ----------
    x : tensor
        points on Hyperboloid
    mean : tensor
        centroid point on Hyperboloid
    k : tensor
        manifold negative curvature
    reducedim : list[int]
        dimensions to reduce. By default all non-manifold dimensions are reduced
    dim : int
        manifold dimension
    keepdim : bool
        retain reduced dimensions

    Returns
    -------
    tensor
        mean squared Lorentz tangent norm
    """
    reducedim = _default_reducedim(x, reducedim, dim)
    u = _logmap(mean, x, k=k, dim=dim)
    sqnorm = _inner(u, u, keepdim=False, dim=dim).clamp_min(0.0)
    reducedim = _reducedim_without_manifold(reducedim, x.dim(), dim)
    if len(reducedim) > 0:
        sqnorm = sqnorm.mean(dim=reducedim, keepdim=keepdim)
    return sqnorm


def _default_reducedim(x, reducedim: Optional[List[int]], dim: int):
    ndim = x.dim()
    dim = dim % ndim
    if reducedim is None:
        reducedim = [d for d in range(ndim) if d != dim]
    else:
        reducedim = sorted(d % ndim for d in reducedim if d % ndim != dim)
    return reducedim


def _reducedim_without_manifold(reducedim: List[int], ndim: int, dim: int):
    dim = dim % ndim
    result = []
    for d in reducedim:
        if d < dim:
            result.append(d)
        elif d > dim:
            result.append(d - 1)
    return result


def _weighted_average(x, weights, reducedim: List[int], dim: int):
    if len(reducedim) == 0:
        return x
    if weights is None or weights.dim() == 0:
        return x.mean(dim=reducedim, keepdim=True)
    dim = dim % x.dim()
    if weights.dim() == x.dim() - 1:
        weights = weights.unsqueeze(dim)
    weights = torch.broadcast_to(weights, x.shape)
    denom = weights.sum(dim=reducedim, keepdim=True).clamp_min(1e-15)
    return (x * weights).sum(dim=reducedim, keepdim=True) / denom


def geodesic_unit(t, x, u, *, k):
    r"""
    Compute unit speed geodesic at time :math:`t` starting from :math:`x` with direction :math:`u/\|u\|_x`.

    .. math::

        \gamma_{\mathbf{x} \rightarrow \mathbf{u}}^{k}(t)=\cosh \left(\frac{t}{\sqrt{k}}\right) \mathbf{x}+\sqrt{k} \sinh \left(\frac{t}{\sqrt{k}}\right) \mathbf{u}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point
    u : tensor
        unit direction vector
    k : tensor
        manifold negative curvature

    Returns
    -------
    tensor
        the point on geodesic line
    """
    return _geodesic_unit(t, x, u, k=k)


@torch.jit.script
def _geodesic_unit(t, x, u, k):
    return (
        torch.cosh(t / torch.sqrt(k)) * x
        + torch.sqrt(k) * torch.sinh(t / torch.sqrt(k)) * u
    )


def half_aperture(x, k, dim=-1, min_radius=0.1, eps=1e-8):
    r"""
    Half-aperture angle of the entailment cone with apex on point x.

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    min_radius : float
        points close to the origin of the Hyperboloid are left with undefined
        half aperture
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        half-aperture of entailment cones with values in `(0, pi/2)`.
    """

    return _half_aperture(x, k, dim, min_radius, eps)


@torch.jit.script
def _half_aperture(x, k, dim: int = -1, min_radius: float = 0.1, eps: float = 1e-8):
    dn = x.size(dim) - 1
    denom = torch.norm(x.narrow(dim, 1, dn), dim=-1) * torch.sqrt(k) + eps
    return torch.asin(torch.clamp(2 * min_radius / denom, min=-1 + eps, max=1 - eps))


def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+\sqrt{k}}

    Parameters
    ----------
    x : tensor
        point on Hyperboloid
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disk
    """
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + torch.sqrt(k))


def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disk to Hyperboloid.

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\sqrt{k} \left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Hyperboloid
    """
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.sqrt(k)
        * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res
