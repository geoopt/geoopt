import pytest
import torch

import geoopt
from geoopt.optim.rlinesearch import LineSearchWarning


def _problem(dtype=torch.float64, geometry="euclidean"):
    manifold = geoopt.Euclidean() if geometry == "euclidean" else geoopt.PoincareBall()
    point = geoopt.ManifoldParameter(torch.tensor([0.2, 0.1, -0.1], dtype=dtype), manifold=manifold)
    reference = geoopt.ManifoldParameter(point.detach().clone(), manifold=manifold)
    target = torch.tensor([0.3, -0.5, 0.2], dtype=dtype)
    weights = torch.tensor([0.7, 1.3, 2.2], dtype=dtype)

    def objective(value):
        return (weights * (value - target).square()).sum()

    return point, reference, objective


def _closure(point, optimizer, objective):
    def evaluate():
        optimizer.zero_grad()
        loss = objective(point)
        loss.backward()
        return loss.item()

    return evaluate


def _compare_step(optimizer, point, reference_optimizer, reference, objective, **kwargs):
    reported = optimizer.step(_closure(point, optimizer, objective), **kwargs)
    reference_optimizer.step(_closure(reference, reference_optimizer, objective))
    torch.testing.assert_close(point, reference)
    assert reported == pytest.approx(objective(point).item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("geometry", ["euclidean", "poincare"])
@pytest.mark.parametrize("compute_derphi", [False, True])
@pytest.mark.parametrize("probe_first", [False, True])
def test_custom_proposals_report_current_loss_and_reuse_current_gradients(
    dtype, geometry, compute_derphi, probe_first
):
    """A callable may return a step directly or reuse one estimated on its first call."""
    point, reference, objective = _problem(dtype, geometry)
    calls = 0

    def choose_step(phi, derphi, **kwargs):
        nonlocal calls
        if probe_first and calls == 0:
            phi(0.1)
        calls += 1
        return 0.1

    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method=choose_step, compute_derphi=compute_derphi
    )
    reference_optimizer = geoopt.optim.RiemannianSGD([reference], lr=0.1)
    for _ in range(3):
        _compare_step(optimizer, point, reference_optimizer, reference, objective)


@pytest.mark.parametrize("compute_derphi", [False, True])
def test_custom_proposal_can_accept_an_earlier_cached_trial(compute_derphi):
    """Accepting an earlier trial keeps its loss while retracting from the current point."""
    point, reference, objective = _problem()

    def choose_step(phi, derphi, **kwargs):
        phi(0.05)
        phi(0.1)
        return 0.05

    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method=choose_step, compute_derphi=compute_derphi
    )
    reference_optimizer = geoopt.optim.RiemannianSGD([reference], lr=0.05)
    for _ in range(3):
        _compare_step(optimizer, point, reference_optimizer, reference, objective)


@pytest.mark.parametrize("compute_derphi", [False, True])
def test_force_step_evaluates_an_unprobed_fallback(compute_derphi):
    """A failed callable can use the documented fallback without missing trial state."""
    point, reference, objective = _problem()
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point],
        line_search_method=lambda phi, derphi, **kwargs: None,
        compute_derphi=compute_derphi,
        fallback_stepsize=0.1,
    )
    reference_optimizer = geoopt.optim.RiemannianSGD([reference], lr=0.1)
    for _ in range(3):
        _compare_step(optimizer, point, reference_optimizer, reference, objective, force_step=True)


@pytest.mark.parametrize("compute_derphi", [False, True])
def test_preview_does_not_reuse_gradients_at_an_uncommitted_point(compute_derphi):
    """A no-step preview leaves the next real update based on the unchanged parameters."""
    point, reference, objective = _problem()
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point],
        line_search_method=lambda phi, derphi, **kwargs: 0.1,
        compute_derphi=compute_derphi,
    )
    reference_optimizer = geoopt.optim.RiemannianSGD([reference], lr=0.1)
    original = point.detach().clone()
    optimizer.step(_closure(point, optimizer, objective), no_step=True)
    assert torch.equal(point, original)
    _compare_step(optimizer, point, reference_optimizer, reference, objective)


@pytest.mark.parametrize("compute_derphi", [False, True])
def test_failed_search_can_retry_from_the_current_point(compute_derphi):
    """A rejected search does not cache gradients for a proposal that never existed."""
    point, reference, objective = _problem()
    calls = 0

    def choose_step(phi, derphi, **kwargs):
        nonlocal calls
        calls += 1
        return None if calls == 1 else 0.1

    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method=choose_step, compute_derphi=compute_derphi
    )
    reference_optimizer = geoopt.optim.RiemannianSGD([reference], lr=0.1)
    original = point.detach().clone()
    with pytest.warns(LineSearchWarning, match="No suitable step size"):
        optimizer.step(_closure(point, optimizer, objective))
    assert torch.equal(point, original)
    _compare_step(optimizer, point, reference_optimizer, reference, objective)
