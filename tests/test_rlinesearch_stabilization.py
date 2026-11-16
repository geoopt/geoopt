import contextlib

import pytest
import torch

import geoopt
from geoopt.optim.rlinesearch import LineSearchWarning


def _problem(geometry):
    if geometry == "sphere":
        manifold = geoopt.Sphere()
        initial = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
        target = torch.tensor([-1.0, 1.0, 2.0], dtype=torch.float64)
    else:
        manifold = geoopt.CanonicalStiefel()
        initial = torch.eye(4, 2, dtype=torch.float64)
        target = torch.tensor(
            [[1.0, -1.0], [2.0, 1.0], [-1.0, 3.0], [2.0, -2.0]],
            dtype=torch.float64,
        )
    point = geoopt.ManifoldParameter(manifold.projx(initial), manifold=manifold)
    target = manifold.projx(target)
    return point, target


def _closure(point, target, optimizer, observations=None):
    def evaluate():
        if observations is not None:
            observations.append(point.detach().clone())
        optimizer.zero_grad()
        loss = (point - target).square().sum()
        loss.backward()
        return loss.item()

    return evaluate


@pytest.mark.parametrize("geometry", ["sphere", "stiefel"])
@pytest.mark.parametrize("method", ["armijo", "wolfe"])
@pytest.mark.parametrize("period", [1, 2, None])
def test_linesearch_periodic_stabilization_preserves_training(geometry, method, period):
    point, target = _problem(geometry)
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method=method, stabilize=period
    )
    initial_loss = (point - target).square().sum().item()
    for _ in range(2):
        optimizer.step(_closure(point, target, optimizer))
        assert point.is_leaf and point.requires_grad
        assert torch.isfinite(point).all()
        assert point.manifold.check_point_on_manifold(point)
    assert (point - target).square().sum().item() < initial_loss
    assert optimizer.param_groups[0]["step"] == 2


@pytest.mark.parametrize("geometry", ["sphere", "stiefel"])
@pytest.mark.parametrize("compute_derphi", [False, True])
@pytest.mark.parametrize("outer_no_grad", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_uncommitted_search_does_not_project_or_advance_stabilization(
    geometry, compute_derphi, outer_no_grad, preview
):
    point, target = _problem(geometry)
    with torch.no_grad():
        point.mul_(1.01)
    original = point.detach().clone()
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point],
        line_search_method=lambda phi, derphi, **kwargs: 0.05 if preview else None,
        compute_derphi=compute_derphi,
        stabilize=1,
    )
    context = torch.no_grad() if outer_no_grad else contextlib.nullcontext()
    warning = contextlib.nullcontext() if preview else pytest.warns(LineSearchWarning)
    with context, warning:
        optimizer.step(_closure(point, target, optimizer), no_step=preview)
    assert torch.equal(point, original)
    assert optimizer.param_groups[0]["step"] == 0
    assert point.is_leaf and point.requires_grad


@pytest.mark.parametrize("geometry", ["sphere", "stiefel"])
@pytest.mark.parametrize("outer_no_grad", [False, True])
def test_manual_stabilization_recomputes_gradient_at_projected_point(
    geometry, outer_no_grad
):
    point, target = _problem(geometry)
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point],
        line_search_method=lambda phi, derphi, **kwargs: 0.05,
        compute_derphi=True,
    )
    observations = []
    closure = _closure(point, target, optimizer, observations)
    optimizer.step(closure)
    with torch.no_grad():
        point.add_(0.01 * target)
        point.mul_(1.01)
        projected = point.manifold.projx(point)
    context = torch.no_grad() if outer_no_grad else contextlib.nullcontext()
    with context:
        optimizer.stabilize()
    torch.testing.assert_close(point, projected)
    assert point.is_leaf and point.requires_grad
    observations.clear()
    expected_gradient = point.manifold.egrad2rgrad(point, 2 * (point - target))
    expected = point.manifold.retr(point, -0.05 * expected_gradient)
    optimizer.step(closure)
    torch.testing.assert_close(observations[0], projected)
    torch.testing.assert_close(point, expected)
