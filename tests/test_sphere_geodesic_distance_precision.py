import math

import pytest
import torch

import geoopt


def circle_points(angles, dtype):
    return torch.tensor([[math.cos(angle), math.sin(angle), 0.0] for angle in angles], dtype=dtype)


def circle_distance(a, b):
    difference = abs(a-b) % (2*math.pi)
    return min(difference, 2*math.pi-difference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-10, 1e-8, 1e-4, -0.2, 0.7, math.pi-1e-7, math.pi])
def test_sphere_distance_retains_endpoints_and_small_arcs(dtype, angle):
    sphere = geoopt.Sphere()
    x, y = circle_points([0.0, angle], dtype)
    expected = torch.tensor(abs(angle), dtype=dtype)
    tolerance = 5e-7 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(sphere.dist(x, y), expected, rtol=tolerance, atol=0)
    torch.testing.assert_close(sphere.dist(y, x), expected, rtol=tolerance, atol=0)
    torch.testing.assert_close(sphere.dist(x, y, keepdim=True), expected[None], rtol=tolerance, atol=0)
    torch.testing.assert_close(sphere.dist(x, x), torch.zeros((), dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pairwise_spherical_distances_match_independent_circle_arcs(dtype):
    sphere = geoopt.Sphere()
    first = [0.0, 1e-8, 0.7, math.pi]
    second = [0.0, -1e-8, 0.2, math.pi]
    x, y = circle_points(first, dtype), circle_points(second, dtype)
    expected = torch.tensor([[circle_distance(a, b) for b in second] for a in first], dtype=dtype)
    actual = sphere.cdist(x, y)
    torch.testing.assert_close(actual, expected, rtol=5e-7 if dtype == torch.float32 else 1e-13, atol=0)
    torch.testing.assert_close(actual, sphere.dist(x[:, None], y[None, :]),
                               rtol=5e-7 if dtype == torch.float32 else 1e-13, atol=0)
    empty = sphere.cdist(x[:0], y)
    assert empty.shape == (0, 4)


def test_pairwise_distance_batch_broadcast_and_direct_kernel_threshold():
    sphere = geoopt.Sphere()
    first = [0.0, 1e-8, 0.7, 1.2]
    second = [0.1, 0.3, 0.8]
    x = torch.stack([circle_points([angle+offset for angle in first], torch.float64)
                     for offset in [0.0, 0.4]])[:, None]
    y = torch.stack([circle_points([angle+offset for angle in second], torch.float64)
                     for offset in [0.0, -0.2, 0.3]])[None]
    expected = torch.tensor([[[[circle_distance(a+oa, b+ob) for b in second] for a in first]
                              for ob in [0.0, -0.2, 0.3]] for oa in [0.0, 0.4]], dtype=torch.float64)
    torch.testing.assert_close(sphere.cdist(x, y), expected, rtol=1e-13, atol=1e-15)
    angles = [index*1e-8 for index in range(31)]
    points = circle_points(angles, torch.float64)
    expected = torch.tensor([[abs(a-b) for b in angles] for a in angles], dtype=torch.float64)
    actual = sphere.cdist(points, points)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=0)
    torch.testing.assert_close(actual.diag(), torch.zeros(31, dtype=torch.float64), rtol=0, atol=0)


def test_small_sphere_arc_has_the_geometric_derivative():
    sphere = geoopt.Sphere()
    origin = circle_points([0.0], torch.float64)[0]
    for value in [1e-10, 1e-8, 0.2]:
        angle = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        point = torch.stack((angle.cos(), angle.sin(), angle.new_zeros(())))
        distance = sphere.dist(origin, point)
        derivative = torch.autograd.grad(distance, angle)[0]
        torch.testing.assert_close(derivative, torch.ones_like(derivative), rtol=1e-12, atol=1e-12)
    angle = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    def distance(theta):
        point = torch.stack((theta.cos(), theta.sin(), theta.new_zeros(())))
        return sphere.dist(origin, point)
    assert torch.autograd.gradcheck(distance, (angle,))
