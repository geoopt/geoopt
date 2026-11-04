import pytest
import torch

import geoopt


MANIFOLDS = [
    geoopt.PoincareBall(c=1.0),
    geoopt.PoincareBallExact(c=0.7),
    geoopt.Stereographic(k=-0.5),
    geoopt.Stereographic(k=0.8),
    geoopt.SphereProjection(k=1.0),
]


@pytest.mark.parametrize("manifold", MANIFOLDS)
@pytest.mark.parametrize("keepdim", [False, True])
def test_inner_matches_conformal_metric(manifold, keepdim):
    torch.manual_seed(0)
    x = manifold.expmap0(0.3 * torch.randn(3, 4, dtype=torch.float64))
    u = torch.randn(3, 4, dtype=torch.float64)
    v = torch.randn(3, 4, dtype=torch.float64)
    lam = manifold.lambda_x(x, keepdim=True)
    expected = (lam**2 * u * v).sum(-1, keepdim=keepdim)

    inner = manifold.inner(x, u, v, keepdim=keepdim)
    assert inner.shape == expected.shape
    torch.testing.assert_close(inner, expected)
    # inner(x, u) is the squared norm
    torch.testing.assert_close(
        manifold.inner(x, u, keepdim=keepdim), manifold.norm(x, u, keepdim=keepdim) ** 2
    )


@pytest.mark.parametrize("manifold", MANIFOLDS)
def test_inner_single_point_is_scalar(manifold):
    x = manifold.expmap0(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
    u = torch.tensor([0.5, 0.1, -0.4], dtype=torch.float64)
    assert manifold.inner(x, u, u).shape == torch.Size([])
    assert manifold.inner(x, u, u, keepdim=True).shape == torch.Size([1])
