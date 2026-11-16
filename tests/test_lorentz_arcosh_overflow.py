import math

import geoopt
import pytest
import torch

from geoopt.manifolds.lorentz.math import arcosh


@pytest.mark.parametrize("value", [1e150, 1e154, 1e155, 1e200, 1e300, 1.7976931348623157e308])
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_arcosh_large_finite_values_and_gradients(value, shape):
    x = torch.full(shape, value, dtype=torch.float64, requires_grad=True)
    actual = arcosh(x)
    expected = torch.full_like(x, math.acosh(value))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=0)
    gradient = torch.autograd.grad(actual.sum(), x)[0]
    inverse = 1.0 / value
    expected_gradient = torch.full_like(x, inverse / math.sqrt(1.0 - inverse * inverse))
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-14, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_arcosh_preserves_regularized_values_and_gradients(dtype):
    x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 16.0, 1e4], dtype=dtype)
    x = torch.cat((x, torch.tensor([1 + torch.finfo(dtype).eps], dtype=dtype)))
    x.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference = torch.log(
        reference_x + torch.sqrt(torch.clamp_min(reference_x.double().square() - 1, 1e-15))
    ).to(dtype)
    actual = arcosh(x)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference)
    observed_gradient = torch.autograd.grad(actual.sum(), x)[0]
    reference_gradient = torch.autograd.grad(reference.sum(), reference_x)[0]
    torch.testing.assert_close(observed_gradient, reference_gradient)


@pytest.mark.parametrize("value", [1.01, 1.99, 2.0, 2.01, 10.0, 1000.0])
def test_arcosh_finite_difference_gradient(value):
    x = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(arcosh, (x,))


@pytest.mark.parametrize("argument", [360.0, 450.0, 600.0])
@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("keepdim", [False, True])
def test_large_lorentz_expmap_distance_round_trip(argument, curvature, dim, keepdim):
    manifold = geoopt.Lorentz(k=torch.tensor(curvature, dtype=torch.float64))
    radius = math.sqrt(curvature)
    distance = torch.tensor(
        [radius * argument, radius * (argument + 1)], dtype=torch.float64, requires_grad=True
    )
    tangent = torch.stack((torch.zeros_like(distance), 0.6 * distance, 0.8 * distance), dim=dim)
    point = manifold.expmap0(tangent, project=False, dim=dim)
    assert torch.isfinite(point).all()
    origin = torch.zeros_like(point)
    origin.select(dim, 0).fill_(radius)
    actual = manifold.dist0(point, dim=dim, keepdim=keepdim)
    pair_distance = manifold.dist(origin, point, dim=dim, keepdim=keepdim)
    expected = distance.unsqueeze(dim) if keepdim else distance
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=0)
    torch.testing.assert_close(pair_distance, expected, rtol=2e-14, atol=0)
    derivative = torch.autograd.grad((actual.sum() + pair_distance.sum()) / 2, distance)[0]
    assert torch.isfinite(derivative).all()
    torch.testing.assert_close(derivative, torch.ones_like(distance), rtol=2e-13, atol=0)
