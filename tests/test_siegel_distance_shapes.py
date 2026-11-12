import math

import pytest
import torch

from geoopt import UpperHalf, BoundedDomain
from geoopt.manifolds.siegel.vvd_metrics import SiegelMetricType


def diagonal_points(manifold_type, batch_shape):
    scale = torch.arange(1, math.prod(batch_shape) + 1, dtype=torch.float64).reshape(
        batch_shape
    )
    values = scale.unsqueeze(-1) * torch.tensor([0.2, 0.4, 0.7], dtype=torch.float64)
    if manifold_type is UpperHalf:
        x = 1j * torch.diag_embed(torch.ones_like(values)).to(torch.complex128)
        y = 1j * torch.diag_embed(values.exp()).to(torch.complex128)
    else:
        x = torch.zeros(batch_shape + (3, 3), dtype=torch.complex128)
        y = torch.diag_embed((values / 2).tanh()).to(torch.complex128)
    return x, y, values


def expected_distance(values, metric):
    if metric is SiegelMetricType.RIEMANNIAN:
        return values.norm(dim=-1)
    if metric is SiegelMetricType.FINSLER_INFINITY:
        return values[..., -1]
    if metric is SiegelMetricType.FINSLER_MINIMUM:
        return 2 * values[..., 1] + 4 * values[..., 2]
    return values.sum(-1)


@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
@pytest.mark.parametrize("metric", list(SiegelMetricType))
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_siegel_distance_preserves_only_requested_dimensions(
    manifold_type, metric, batch_shape, keepdim
):
    manifold = manifold_type(metric=metric, rank=3)
    x, y, values = diagonal_points(manifold_type, batch_shape)
    expected = expected_distance(values, metric)
    if keepdim:
        expected = expected.reshape(batch_shape + (1, 1))
    actual = manifold.dist(x, y, keepdim=keepdim)
    torch.testing.assert_close(actual, expected)
    if metric is SiegelMetricType.WEIGHTED_SUM:
        actual.sum().backward()
        expected_gradient = (
            values.reshape(-1, 3).sum(0, keepdim=True).to(manifold.metric.weights)
        )
        torch.testing.assert_close(manifold.metric.weights.grad, expected_gradient)
        assert manifold.state_dict()["metric.weights"].shape == (1, 3)


@pytest.mark.parametrize("manifold_type", [UpperHalf, BoundedDomain])
@pytest.mark.parametrize("metric", list(SiegelMetricType)[1:])
def test_default_finsler_distance_does_not_cross_broadcast_training_targets(
    manifold_type, metric
):
    manifold = manifold_type(metric=metric, rank=3)
    x, y, values = diagonal_points(manifold_type, (3,))
    targets = expected_distance(values, metric)
    loss = (manifold.dist(x, y) - targets).square().mean()
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-20, rtol=0)
