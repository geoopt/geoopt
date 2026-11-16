import pytest
import torch

import geoopt


def _symmetric(value):
    return (value + value.transpose(-1, -2)) / 2


def _point(dtype, batch_shape):
    value = torch.tensor(
        [[0.1 + 0.3j, 0.2 + 0.1j], [0.2 + 0.1j, -0.3 + 0.2j]],
        dtype=dtype,
    )
    return value.expand(*batch_shape, 2, 2).clone()


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_gradient_represents_directional_derivatives(dtype, batch_shape):
    manifold = geoopt.BoundedDomain()
    point = _point(dtype, batch_shape).requires_grad_()
    target = torch.tensor([[1 + 2j, 3 - 1j], [3 - 1j, -2 + 4j]], dtype=dtype)
    loss = (point - target).abs().square().sum()
    (gradient,) = torch.autograd.grad(loss, point)
    original = gradient.clone()
    rgradient = manifold.egrad2rgrad(point, gradient)

    assert manifold.check_point_on_manifold(point)
    torch.testing.assert_close(rgradient, rgradient.transpose(-1, -2))
    torch.testing.assert_close(gradient, original, rtol=0, atol=0)

    for component in (1, 1j):
        for row, column in ((0, 0), (0, 1), (1, 1)):
            direction = torch.zeros_like(point)
            direction[..., row, column] = component
            direction[..., column, row] = component
            expected = (gradient.conj() * direction).real.sum(dim=(-1, -2))
            actual = manifold.inner(point, rgradient, direction).real
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_gradient_accepts_broadcast_views_and_preserves_inputs(dtype):
    manifold = geoopt.BoundedDomain()
    point = _point(dtype, (2, 1))
    raw = torch.arange(12, dtype=point.real.dtype).reshape(3, 2, 2)
    gradient = _symmetric(torch.complex(raw, raw.flip(-1))).expand(2, 3, 2, 2)
    point_before = point.clone()
    gradient_before = gradient.clone()
    rgradient = manifold.egrad2rgrad(point, gradient)
    direction = torch.ones_like(rgradient) * (1 - 2j)

    expected = (gradient.conj() * direction).real.sum(dim=(-1, -2))
    actual = manifold.inner(point, rgradient, direction).real
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(point, point_before, rtol=0, atol=0)
    torch.testing.assert_close(gradient, gradient_before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_diagonal_points_keep_the_existing_gradient(dtype):
    manifold = geoopt.BoundedDomain()
    point = torch.diag(torch.tensor([0.1 + 0.2j, -0.3 + 0.1j], dtype=dtype))
    gradient = torch.tensor([[1 + 2j, 3 - 1j], [3 - 1j, -2 + 4j]], dtype=dtype)
    factors = 1 - point.diagonal().abs().square()
    expected = factors.unsqueeze(-1) * gradient * factors.unsqueeze(-2)

    torch.testing.assert_close(manifold.egrad2rgrad(point, gradient), expected)


def test_sgd_step_has_the_metric_dual_initial_velocity():
    manifold = geoopt.BoundedDomain()
    point = geoopt.ManifoldParameter(_point(torch.complex128, ()), manifold=manifold)
    previous = point.detach().clone()
    target = torch.tensor([[1 + 2j, 3 - 1j], [3 - 1j, -2 + 4j]], dtype=point.dtype)
    optimizer = geoopt.optim.RiemannianSGD([point], lr=1e-4)
    (point - target).abs().square().sum().backward()
    gradient = point.grad.detach().clone()
    optimizer.step()

    velocity = (previous - point.detach()) / 1e-4
    direction = torch.tensor([[2 + 1j, -1 + 3j], [-1 + 3j, 4 - 2j]], dtype=point.dtype)
    expected = (gradient.conj() * direction).real.sum()
    torch.testing.assert_close(manifold.inner(previous, velocity, direction).real, expected)
    assert manifold.check_point_on_manifold(point)
    torch.testing.assert_close(point.grad, gradient, rtol=0, atol=0)
