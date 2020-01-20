import torch.jit
import torch.tensor
from typing import List, Optional
from .math import _lambda_x, _mobius_scalar_mul, _antipode, _dist
from ...utils import list_range, drop_dims


@torch.jit.script
def weighted_midpoint(
    xs: torch.Tensor,
    k: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reducedim: Optional[List[int]] = None,
    dim: int = -1,
    keepdim: bool = False,
    lincomb: bool = False,
):
    r"""
    Compute weighted Möbius gyromidpoint.

    The weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights
    :math:`\alpha_1,...,\alpha_n` is computed as follows:

    The weighted Möbius gyromidpoint is computed as follows

    .. math::

        m_{\kappa}(x_1,\ldots,x_n,\alpha_1,\ldots,\alpha_n)
        =
        \frac{1}{2}
        \otimes_\kappa
        \left(
        \sum_{i=1}^n
        \frac{
        \alpha_i\lambda_{x_i}^\kappa
        }{
        \sum_{j=1}^n\alpha_j(\lambda_{x_j}^\kappa-1)
        }
        x_i
        \right)

    where the weights :math:`\alpha_1,...,\alpha_n` do not necessarily need
    to sum to 1 (only their relative weight matters). Note that this formula
    also requires to choose between the midpoint and its antipode for
    :math:`\kappa > 0`.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list|tuple
        reduce dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    k : tensor
        constant sectional curvature
    keepdim : bool
        retain the last dim? (default: false)
    lincomb : bool
        linear combination implementation

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    """
    if reducedim is None:
        reducedim = list_range(xs.dim())
        reducedim.pop(dim)
    gamma = _lambda_x(xs, k=k, dim=dim, keepdim=True)
    if weights is None:
        weights = torch.tensor(1.0, dtype=xs.dtype, device=xs.device)
    else:
        weights = weights.unsqueeze(dim)
    nominator = (gamma * weights * xs).sum(reducedim, keepdim=True)
    denominator = ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    two_mean = nominator / denominator
    two_mean = torch.where(torch.isfinite(two_mean), two_mean, two_mean.new_zeros(()))
    if lincomb and weights.numel() == 1:
        alpha = torch.tensor(0.5, dtype=xs.dtype, device=xs.device) * weights
        for d in reducedim:
            alpha *= xs.size(d)
    elif lincomb:
        alpha = 0.5 * weights.sum(reducedim, keepdim=True)
    else:
        alpha = torch.tensor(0.5, dtype=xs.dtype, device=xs.device)
    mean = _mobius_scalar_mul(alpha, two_mean, k=k, dim=dim)
    if torch.any(k.gt(0)):
        # check antipode
        a_mean = _antipode(mean, k, dim=dim)
        dist = _dist(mean, xs, k=k, keepdim=True, dim=dim).sum(reducedim, keepdim=True)
        a_dist = _dist(a_mean, xs, k=k, keepdim=True, dim=dim).sum(reducedim, keepdim=True)
        better = k.gt(0) & (a_dist < dist)
        mean = torch.where(better, a_mean, mean)
    if not keepdim:
        mean = drop_dims(mean, reducedim)
    return mean
