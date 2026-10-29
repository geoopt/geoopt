import pytest
import torch
from geoopt.manifolds.lorentz import math


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("distance", [0.0, 1e-5, 0.3])
def test_geodesic_transport(dtype, dim, curvature, distance):
    k = torch.tensor(curvature, dtype=dtype)
    r = k.sqrt()
    s = torch.tensor([distance, -distance], dtype=dtype)
    x = torch.stack((r.expand_as(s), s * 0, s * 0), -1).movedim(-1, dim)
    y = torch.stack((r * s.cosh(), r * s.sinh(), s * 0), -1).movedim(-1, dim)
    v = torch.stack((s * 0, torch.ones_like(s), s * 0), -1).movedim(-1, dim)
    expected = torch.stack((s.sinh(), s.cosh(), s * 0), -1).movedim(-1, dim)
    for actual in (
        math.parallel_transport(x, y, v, k=k, dim=dim),
        math.parallel_transport0(y, v, k=k, dim=dim),
    ):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(
            math.inner(y, actual, dim=dim), torch.zeros_like(s), atol=1e-7, rtol=0
        )
        torch.testing.assert_close(
            math.inner(actual, actual, dim=dim), torch.ones_like(s), atol=1e-6, rtol=1e-6
        )
    for actual in (
        math.parallel_transport(y, x, expected, k=k, dim=dim),
        math.parallel_transport0back(y, expected, k=k, dim=dim),
    ):
        torch.testing.assert_close(actual, v, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
@pytest.mark.parametrize("kind", ["general", "origin", "back"])
def test_endpoint_derivative(dtype, curvature, kind):
    k = torch.tensor(curvature, dtype=dtype)
    r = k.sqrt()
    x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype) * r
    v = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)

    def transport(s):
        y = torch.stack((r * s.cosh(), r * s.sinh(), s * 0))
        if kind == "general":
            return math.parallel_transport(x, y, v, k=k)
        if kind == "origin":
            return math.parallel_transport0(y, v, k=k)
        # A tangent vector orthogonal to x in the ambient metric, transported back.
        tangent = torch.stack((s.sinh(), s.cosh(), s * 0))
        return math.parallel_transport0back(y, tangent, k=k)

    t = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    derivative = torch.autograd.functional.jacobian(transport, t)
    expected = torch.tensor([0.0 if kind == "back" else 1.0, 0.0, 0.0], dtype=dtype)
    torch.testing.assert_close(derivative, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
@pytest.mark.parametrize("step", [0.0, 1e-7, 0.4])
def test_non_origin_transport(dim, curvature, step):
    k = torch.tensor(curvature, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([0.7, -0.8], dtype=torch.float64)
    b = a + step

    def point(t):
        return (k.sqrt() * torch.stack((t.cosh(), t.sinh(), t * 0), -1)).movedim(-1, dim)

    def tangent(t):
        return torch.stack((t.sinh(), t.cosh(), torch.ones_like(t)), -1).movedim(-1, dim)

    actual = math.parallel_transport(point(a), point(b), tangent(a), k=k, dim=dim)
    torch.testing.assert_close(actual, tangent(b), atol=1e-12, rtol=1e-12)
    derivative = torch.autograd.grad(actual.sum(), k)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(k), atol=1e-12, rtol=0)
