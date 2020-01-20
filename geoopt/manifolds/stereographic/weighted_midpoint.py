import torch
from .math import _lambda_x
from ...utils import prod, drop_dims, reduce_dim


def weighted_midpoint(
    xs, weights=None, *, ball, reducedim=None, dim=-1, keepdim=False, lincomb=False
):
    """
    Computes the weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights :math:`\alpha_1,...,\alpha_n`.

    The gyromidpoint looks as follows:

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
        average dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    ball : geoopt.Manifold
        Poincare Ball
    keepdim : bool
        retain the last dim? (default: false)
    lincomb : bool
        linear combination implementation

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    """
    if torch.any(ball.k.ge(0)):
        raise RuntimeError("not yet supported k>0")
    reducedim = reduce_dim(xs.dim(), reducedim, dim)
    gamma = ball.lambda_x(xs, dim=dim, keepdim=True)
    if weights is None:
        weights = 1.0
    else:
        weights = weights.unsqueeze(dim)
    nominator = (gamma * weights * xs).sum(reducedim, keepdim=True)
    denominator = ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    two_mean = nominator / denominator
    two_mean = torch.where(torch.isfinite(two_mean), two_mean, two_mean.new_zeros(()))
    if lincomb and isinstance(weights, float):
        alpha = 0.5 * prod((xs.shape[i] for i in reducedim))
    elif lincomb:
        alpha = 0.5 * weights.sum(reducedim, keepdim=True)
    else:
        alpha = 0.5
    mean = ball.mobius_scalar_mul(alpha, two_mean, dim=dim)
    if not keepdim:
        mean = drop_dims(mean, reducedim)
    return mean

