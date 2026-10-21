import math

import pytest
import torch

import geoopt
from geoopt.manifolds.stereographic.math import arsinh


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arsinh_values_and_gradients(dtype):
    x = torch.tensor(
        [-1e8, -1e4, -1.0, -1e-8, 0.0, 1e-8, 1.0, 1e4, 1e8],
        dtype=dtype,
        requires_grad=True,
    )
    expected = torch.tensor([math.asinh(value) for value in x.tolist()], dtype=dtype)
    expected_grad = torch.tensor([1 / math.hypot(value, 1.0) for value in x.tolist()], dtype=dtype)

    result = arsinh(x)
    gradient = torch.autograd.grad(result.sum(), x)[0]

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(gradient, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_hyperplane_distance_and_gradient(dtype):
    ball = geoopt.PoincareBall().to(dtype=dtype)
    x = torch.tensor([[-0.99, 0.0]], dtype=dtype, requires_grad=True)
    p = torch.tensor([[0.99, 0.0]], dtype=dtype)
    normal = torch.tensor([[1.0, 0.0]], dtype=dtype)
    assert ball.check_point_on_manifold(x)
    assert ball.check_point_on_manifold(p)

    signed = ball.dist2plane(x, p, normal, signed=True)
    unsigned = ball.dist2plane(x, p, normal)
    gradient = torch.autograd.grad(signed.sum(), x)[0]

    # On a diameter, this hyperplane is reached at p. The signed distance
    # follows directly by integrating the Poincare line element 2/(1-c*r^2).
    c = ball.c.item()
    sqrt_c = math.sqrt(c)
    x_coord, p_coord = x[0, 0].item(), p[0, 0].item()
    expected = 2 / sqrt_c * (math.atanh(sqrt_c * x_coord) - math.atanh(sqrt_c * p_coord))
    expected_grad = torch.tensor([[2 / (1 - c * x_coord**2), 0.0]], dtype=dtype)
    # Mobius addition near the boundary has a separate float32 rounding error.
    rtol = 2e-3 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(signed, -unsigned)
    torch.testing.assert_close(signed, torch.tensor([expected], dtype=dtype), rtol=rtol, atol=0)
    torch.testing.assert_close(gradient, expected_grad, rtol=rtol, atol=0)
