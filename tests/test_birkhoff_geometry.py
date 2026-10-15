import numpy as np
import pytest
import torch

import geoopt


@pytest.fixture
def manifold():
    return geoopt.BirkhoffPolytope()


@pytest.fixture
def point():
    return torch.tensor([[.6, .3, .1], [.1, .5, .4], [.3, .2, .5]], dtype=torch.float64)


@pytest.fixture
def tangent():
    return torch.tensor([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]], dtype=torch.float64)


def test_projection_preserves_existing_tangent(manifold, point, tangent):
    torch.testing.assert_close(manifold.proju(point, tangent), tangent)
    torch.testing.assert_close(manifold.transp(point, point, tangent), tangent)


def test_uniform_point_projection(manifold, tangent):
    point = torch.ones(3, 3, dtype=torch.float64) / 3
    torch.testing.assert_close(manifold.proju(point, tangent), tangent)


def test_projection_matches_constrained_weighted_least_squares(manifold, point):
    ambient = torch.arange(9, dtype=torch.float64).reshape(3, 3).square() / 7
    n = point.shape[-1]
    # Impose every row sum and all but one column sum: an independent,
    # full-rank representation of the tangent constraints.
    constraints = np.zeros((2 * n - 1, n * n))
    for row in range(n):
        constraints[row, row * n:(row + 1) * n] = 1
    for col in range(n - 1):
        constraints[n + col, col::n] = 1
    weights = np.diag(point.numpy().reshape(-1))
    multipliers = np.linalg.solve(constraints @ weights @ constraints.T,
                                  constraints @ ambient.numpy().reshape(-1))
    expected = ambient.numpy().reshape(-1) - weights @ constraints.T @ multipliers
    actual = manifold.proju(point, ambient)
    torch.testing.assert_close(actual, torch.from_numpy(expected.reshape(n, n)))
    torch.testing.assert_close(actual.sum(-1), torch.zeros(n, dtype=point.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(actual.sum(-2), torch.zeros(n, dtype=point.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(manifold.proju(point, actual), actual)


def test_projection_residual_is_metric_normal(manifold, point, tangent):
    ambient = torch.arange(9, dtype=torch.float64).reshape(3, 3).square() / 7
    projected = manifold.proju(point, ambient)
    torch.testing.assert_close(manifold.inner(point, ambient - projected, tangent),
                               torch.tensor(0., dtype=point.dtype), atol=1e-12, rtol=0)


def test_riemannian_gradient_represents_directional_derivative(manifold, point, tangent):
    ambient = torch.arange(9, dtype=torch.float64).reshape(3, 3).square() / 7
    gradient = manifold.egrad2rgrad(point, ambient)
    actual = manifold.inner(point, gradient, tangent)
    expected = (ambient * tangent).sum()
    torch.testing.assert_close(actual, expected)


def test_batched_projection_and_gradient(manifold, point, tangent):
    points = torch.stack((point, torch.ones_like(point) / 3))
    projected = manifold.proju(points, tangent)
    torch.testing.assert_close(projected, tangent.expand_as(points))
    ambient = torch.arange(9, dtype=torch.float64).reshape(3, 3).square() / 7
    gradient = manifold.egrad2rgrad(points, ambient)
    actual = manifold.inner(points, gradient, tangent)
    torch.testing.assert_close(actual, (ambient * tangent).sum().expand(2))
