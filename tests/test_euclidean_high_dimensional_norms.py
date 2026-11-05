import math

import pytest
import torch

import geoopt


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_euclidean_norm_and_distance_reduce_all_manifold_dimensions(ndim, keepdim, dtype):
    manifold = geoopt.Euclidean(ndim=ndim)
    shape = (2,) * ndim
    values = [float(i + 1) for i in range(math.prod(shape))]
    tangent = torch.tensor(values, dtype=dtype).reshape(shape)
    tangent = torch.stack((tangent, 2 * tangent)).requires_grad_()
    point = torch.full_like(tangent, 7)
    expected = torch.tensor(
        [math.sqrt(sum(value * value for value in values)),
         2 * math.sqrt(sum(value * value for value in values))],
        dtype=dtype,
    )
    if keepdim:
        expected = expected.reshape((2,) + (1,) * ndim)

    norm = manifold.norm(point, tangent, keepdim=keepdim)
    distance = manifold.dist(point, point + tangent, keepdim=keepdim)

    assert norm.shape == expected.shape
    assert distance.shape == expected.shape
    torch.testing.assert_close(norm, expected)
    torch.testing.assert_close(distance, expected)
    expected_gradient = tangent.detach() / expected.reshape((2,) + (1,) * ndim)
    for result in (norm, distance):
        gradient = torch.autograd.grad(result.sum(), tangent, retain_graph=True)[0]
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("ndim", [3, 4])
def test_high_dimensional_euclidean_zero_norm_has_finite_zero_gradient(ndim):
    manifold = geoopt.Euclidean(ndim=ndim)
    tangent = torch.zeros((2,) * ndim, dtype=torch.float64, requires_grad=True)
    point = torch.ones_like(tangent)

    for result in (manifold.norm(point, tangent), manifold.dist(point, point + tangent)):
        torch.testing.assert_close(result, result.new_zeros(()))
        gradient = torch.autograd.grad(result, tangent, retain_graph=True)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(tangent))
