import geoopt
import pytest
import torch


def reference_expmap(x, u):
    columns = x.shape[-1]
    xtu = x.transpose(-1, -2) @ u
    utu = u.transpose(-1, -2) @ u
    identity = torch.eye(columns, dtype=x.dtype).expand_as(utu)
    generator = torch.cat(
        (torch.cat((xtu, -utu), dim=-1), torch.cat((identity, xtu), dim=-1)), dim=-2
    )
    selector = torch.cat((torch.matrix_exp(-xtu), torch.zeros_like(utu)), dim=-2)
    return torch.cat((x, u), dim=-1) @ torch.matrix_exp(generator) @ selector


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("shape", [(6, 3), (32, 4), (4, 4)])
def test_expmap_block_evaluation_matches_full_formula(dtype, batch, shape):
    torch.manual_seed(1738)
    x = torch.linalg.qr(torch.randn(*batch, *shape, dtype=dtype)).Q
    manifold = geoopt.EuclideanStiefel()
    u = manifold.proju(x, 0.1 * torch.randn_like(x))
    expected = reference_expmap(x, u)
    actual = manifold.expmap(x, u)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    identity = torch.eye(shape[-1], dtype=dtype).expand(*batch, shape[-1], shape[-1])
    torch.testing.assert_close(
        actual.transpose(-1, -2) @ actual, identity, rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("cls", [geoopt.EuclideanStiefel, geoopt.EuclideanStiefelExact])
def test_exact_curve_initial_velocity_and_zero_step(cls):
    torch.manual_seed(1739)
    manifold = cls()
    x = torch.linalg.qr(torch.randn(9, 3, dtype=torch.float64)).Q
    u = manifold.proju(x, torch.randn_like(x))
    h = 1e-6
    derivative = (manifold.expmap(x, h * u) - manifold.expmap(x, -h * u)) / (2 * h)
    torch.testing.assert_close(derivative, u, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(manifold.expmap(x, torch.zeros_like(u)), x)
    if cls is geoopt.EuclideanStiefelExact:
        torch.testing.assert_close(
            manifold.retr(x, 0.1 * u), reference_expmap(x, 0.1 * u)
        )


def test_expmap_derivatives_of_point_and_tangent():
    torch.manual_seed(1740)
    manifold = geoopt.EuclideanStiefel()
    x = torch.linalg.qr(torch.randn(4, 2, dtype=torch.float64)).Q
    u = manifold.proju(x, 0.1 * torch.randn_like(x))
    x.requires_grad_()
    u.requires_grad_()
    assert torch.autograd.gradcheck(manifold.expmap, (x, u))
    weight = torch.randn_like(x)
    actual = torch.autograd.grad((manifold.expmap(x, u) * weight).sum(), (x, u))
    expected = torch.autograd.grad((reference_expmap(x, u) * weight).sum(), (x, u))
    for derivative, reference in zip(actual, expected):
        torch.testing.assert_close(derivative, reference, rtol=1e-8, atol=1e-8)
