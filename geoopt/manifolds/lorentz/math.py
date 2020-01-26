import torch.jit
import torch as th


@torch.jit.script
def arcosh(x: torch.Tensor):
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-15))
    return torch.log(x + z)


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
    return _inner0(v, keepdim=keepdim, dim=dim)


@torch.jit.script
def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1) * torch.sqrt(k)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def dist(x, y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid.

    .. math::

        d_{\mathcal{L}}^{k}(\mathbf{x}, \mathbf{y})=\sqrt{k} \operatorname{arcosh}\left(-\langle\mathbf{x}, \mathbf{y}\rangle_{\mathcal{L}} / k\right)

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


def dist0(y, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero point

    .. math::

    Parameters
    ----------
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
    return _dist0(y, k=k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist0(y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(y, k=k, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def dist0(x, *, k, keepdim=False, dim=-1):
    r"""
    Compute geodesic distance on the Hyperboloid to zero.

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
        geodesic distance between :math:`x` and :math:`0`
    """
    return _dist0(x, k, keepdim=keepdim, dim=dim)


@torch.jit.script
def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    zp = torch.ones_like(x)
    d = zp.size(dim) - 1
    zp = torch.cat(
        (zp.narrow(dim, 0, 1) * torch.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )
    d = -_inner(x, zp, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)


def project(x, *, k, dim=-1):
    r"""
    Projection on the Hyperboloid.

    .. math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathbb{H}^{d, 1}}(\mathbf{x}):=\left(\sqrt{1+\left\|\mathbf{x}_{1: d}\right\|_{2}^{2}}, \mathbf{x}_{1: d}\right)

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
        \pi((\mathbf{d}, r))=(\sinh (r) \mathbf{d}, \cosh (r))

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
    Projection of the vector on the tangent space of the Hyperboloid.

    ... math::

        \Pi_{\mathbb{R}^{d+1} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{H}^{d, 1}(\mathbf{v})}:=\mathbf{v}+\langle\mathbf{x}, \mathbf{v}\rangle_{\mathcal{L}} \mathbf{x}

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
    return torch.sqrt(_inner(u, u, keepdim=keepdim))


def expmap(x, u, *, k, dim=-1):
    r"""
    Compute exponential map on the Hyperboloid.

    .. math::

        \exp _{\mathbf{x}}^{K}(\mathbf{v})=\cosh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{K}}\right) \mathbf{x}+\sqrt{K} \sinh \left(\frac{\|\mathbf{v}\|_{\mathcal{L}}}{\sqrt{K}}\right) \frac{\mathbf{v}}{\|\mathbf{v}\|_{\mathcal{L}}}


    Parameters
    ----------
    x : tensor
        starting point on Hyperboloid
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
    zp = torch.ones_like(u)
    d = zp.size(dim) - 1
    zp = torch.cat(
        (zp.narrow(dim, 0, 1) * torch.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )
    nomin = _norm(u, keepdim=True, dim=dim)
    p = (
        torch.cosh(nomin / torch.sqrt(k)) * zp
        + torch.sqrt(k) * torch.sinh(nomin / torch.sqrt(k)) * u / nomin
    )
    return p


def logmap(x, y, *, k, dim=-1):
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
    zp = torch.ones_like(y)
    d = zp.size(dim) - 1
    zp = torch.cat(
        (zp.narrow(dim, 0, 1) * torch.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )

    dist_ = _dist0(y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * _inner0(y, k=k, keepdim=True) * zp
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
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True).expand_as(x), x / k)
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
    zp = torch.ones_like(y)
    d = zp.size(dim) - 1
    zp = torch.cat(
        (zp.narrow(dim, 0, 1) * torch.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )
    lmap = _logmap0(y, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(y, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap(y, zp, k=k, dim=dim))
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
    # TODO: zp tensor creation should be removed
    zp = torch.ones_like(x)
    d = zp.size(dim) - 1
    zp = torch.cat(
        (zp.narrow(dim, 0, 1) * torch.sqrt(k), zp.narrow(dim, 1, d) * 0.0), dim=dim
    )
    # TODO: logmap0back is required
    lmap = _logmap0(x, k=k, dim=dim)
    nom = _inner(lmap, v, keepdim=True)
    denom = _dist0(x, k=k, dim=dim, keepdim=True) ** 2
    p = v - nom / denom * (lmap + _logmap0(x, k=k, dim=dim))
    return p


def geodesic_unit(t, x, u, *, k):
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


def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disk.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+1}

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

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

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
