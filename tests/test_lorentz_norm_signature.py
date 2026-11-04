import torch

import geoopt


def _point_and_tangent(manifold):
    torch.manual_seed(0)
    x = manifold.random_normal(3, 4, std=0.5, dtype=torch.float64)
    u = manifold.proju(x, torch.randn(3, 4, dtype=torch.float64))
    return x, u


def test_norm_accepts_point_and_vector():
    manifold = geoopt.Lorentz(k=torch.tensor(1.5, dtype=torch.float64))
    x, u = _point_and_tangent(manifold)
    expected = manifold.inner(x, u, u).sqrt()
    torch.testing.assert_close(manifold.norm(x, u), expected)
    torch.testing.assert_close(manifold.norm(x, u, keepdim=True), expected[..., None])
    # the historical single-argument form keeps working
    torch.testing.assert_close(manifold.norm(u), expected)


def test_scaled_lorentz_norm():
    manifold = geoopt.Lorentz(k=torch.tensor(1.0, dtype=torch.float64))
    scaled = geoopt.Scaled(manifold, 2.0)
    x, u = _point_and_tangent(manifold)
    norm = scaled.norm(x, u)
    torch.testing.assert_close(norm, scaled.inner(x, u, u).sqrt())
    torch.testing.assert_close(scaled.dist(x, scaled.expmap(x, u)), norm)
