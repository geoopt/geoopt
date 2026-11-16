import geoopt
import pytest
import torch


def dense_reference(x, u):
    return u - (x @ u.transpose(-1, -2)) @ x


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [(7, 3), (5, 5), (32, 2)])
@pytest.mark.parametrize("batches", [((), ()), ((2,), (2,)), ((2, 1), (1, 3)), ((), (0,))])
def test_canonical_gradient_values_and_broadcast(dtype, size, batches):
    torch.manual_seed(1910)
    x = torch.linalg.qr(torch.randn(*batches[0], *size, dtype=dtype)).Q
    u = torch.randn(*batches[1], *size, dtype=dtype)
    actual = geoopt.CanonicalStiefel().egrad2rgrad(x, u)
    expected = dense_reference(x, u)
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    tangent_constraint = x.transpose(-1, -2) @ actual + actual.transpose(-1, -2) @ x
    torch.testing.assert_close(tangent_constraint, torch.zeros_like(tangent_constraint), rtol=0, atol=3e-6)


@pytest.mark.parametrize("size", [(7, 3), (5, 5), (32, 2)])
def test_canonical_gradient_metric_duality(size):
    torch.manual_seed(1911)
    manifold = geoopt.CanonicalStiefel()
    x = torch.linalg.qr(torch.randn(2, *size, dtype=torch.float64)).Q
    u = torch.randn_like(x)
    tangent = manifold.proju(x, torch.randn_like(x))
    gradient = manifold.egrad2rgrad(x, u)
    torch.testing.assert_close(manifold.inner(x, gradient, tangent), (u * tangent).sum((-2, -1)))


def test_canonical_gradient_backward_and_noncontiguous_inputs():
    torch.manual_seed(1912)
    manifold = geoopt.CanonicalStiefel()
    x = torch.randn(2, 3, 5, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    u = torch.randn(2, 3, 5, dtype=torch.float64).transpose(-1, -2).requires_grad_()
    assert not x.is_contiguous()
    assert not u.is_contiguous()
    assert torch.autograd.gradcheck(manifold.egrad2rgrad, (x, u))
    weight = torch.randn_like(x)
    actual = torch.autograd.grad((manifold.egrad2rgrad(x, u) * weight).sum(), (x, u))
    expected = torch.autograd.grad((dense_reference(x, u) * weight).sum(), (x, u))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


class ReferenceCanonicalStiefel(geoopt.CanonicalStiefel):
    def egrad2rgrad(self, x, u):
        return dense_reference(x, u)


@pytest.mark.parametrize("optimizer_class", [geoopt.optim.RiemannianSGD, geoopt.optim.RiemannianAdam])
def test_canonical_gradient_optimizer_trajectory(optimizer_class):
    torch.manual_seed(1913)
    manifold = geoopt.CanonicalStiefel()
    initial = torch.linalg.qr(torch.randn(15, 4, dtype=torch.float64)).Q
    point = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    reference = geoopt.ManifoldParameter(initial.clone(), manifold=ReferenceCanonicalStiefel())
    target = torch.randn_like(initial)
    optimizer = optimizer_class([point], lr=0.01)
    reference_optimizer = optimizer_class([reference], lr=0.01)
    for _ in range(6):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        (point - target).square().sum().backward()
        (reference - target).square().sum().backward()
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(point, reference, rtol=1e-10, atol=1e-11)
