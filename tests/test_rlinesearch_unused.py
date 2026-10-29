import copy

import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize(
    "method,cg",
    [("armijo", "steepest"), ("armijo", "pr"), ("wolfe", "fr"), ("wolfe", "pr")],
)
def test_linesearch_ignores_parameters_without_gradients(dtype, frozen, method, cg):
    """An unused parameter does not alter the active-only line search or conjugate history."""
    point = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0, -0.5, 1.25], dtype=dtype))
    reference = torch.nn.Parameter(point.detach().clone())
    unused = torch.nn.Parameter(torch.tensor([7.0, -3.0], dtype=dtype), requires_grad=not frozen)
    original = unused.detach().clone()
    weights = torch.tensor([0.7, 1.3, 2.2, 3.1, 5.0], dtype=dtype)
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point, unused], line_search_method=method, cg_method=cg
    )
    reference_optimizer = geoopt.optim.RiemannianLineSearch(
        [reference], line_search_method=method, cg_method=cg
    )

    def closure():
        optimizer.zero_grad()
        loss = (weights * (point - 0.3).square()).sum()
        loss.backward()
        return loss.item()

    def reference_closure():
        reference_optimizer.zero_grad()
        loss = (weights * (reference - 0.3).square()).sum()
        loss.backward()
        return loss.item()

    for _ in range(3):
        actual = optimizer.step(closure)
        expected = reference_optimizer.step(reference_closure)
        assert actual == pytest.approx(expected)
        torch.testing.assert_close(point, reference)
        assert unused.grad is None
        assert unused not in optimizer.state
        assert torch.equal(unused, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["armijo", "wolfe"])
def test_linesearch_finetunes_a_head_with_a_frozen_backbone(dtype, method):
    """Passing model.parameters() works for a small network with a frozen first layer."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        ).to(dtype)
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    reference = copy.deepcopy(model)
    frozen_before = [p.detach().clone() for p in model[0].parameters()]
    inputs = torch.tensor([[1.0, -0.5, 0.25], [0.5, 1.0, -2.0], [0.0, 2.0, 0.7]], dtype=dtype)
    optimizer = geoopt.optim.RiemannianLineSearch(model.parameters(), line_search_method=method)
    reference_optimizer = geoopt.optim.RiemannianLineSearch(
        [p for p in reference.parameters() if p.requires_grad], line_search_method=method
    )

    def closure():
        optimizer.zero_grad()
        loss = model(inputs).square().mean()
        loss.backward()
        return loss.item()

    def reference_closure():
        reference_optimizer.zero_grad()
        loss = reference(inputs).square().mean()
        loss.backward()
        return loss.item()

    for _ in range(3):
        assert optimizer.step(closure) == pytest.approx(reference_optimizer.step(reference_closure))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
    for actual, expected in zip(model[0].parameters(), frozen_before):
        assert torch.equal(actual, expected)
        assert actual.grad is None
        assert actual not in optimizer.state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cg", ["fr", "pr"])
def test_linesearch_restarts_direction_when_recomputed_active_set_changes(dtype, cg):
    """Freeze/unfreeze with fresh gradients restarts conjugacy for the changed search space."""
    left = torch.nn.Parameter(torch.tensor([-1.0, 2.0], dtype=dtype))
    right = torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=dtype))

    def fixed_step(phi, derphi, **kwargs):
        phi(0.05)
        derphi(0.05)
        return 0.05

    optimizer = geoopt.optim.RiemannianLineSearch(
        [left, right], line_search_method=fixed_step, cg_method=cg
    )

    def loss_value():
        return ((left + 2 * right - 0.3).square() * torch.tensor([0.7, 1.3], dtype=dtype)).sum()

    def closure():
        optimizer.zero_grad()
        loss = loss_value()
        loss.backward()
        return loss.item()

    for train_right in (True, False, True):
        right.requires_grad_(train_right)
        active = [p for p in (left, right) if p.requires_grad]
        gradients = torch.autograd.grad(loss_value(), active)
        expected = [p.detach() - 0.05 * gradient for p, gradient in zip(active, gradients)]
        frozen_right = right.detach().clone()
        optimizer.step(closure, recompute_gradients=True)
        for actual, reference in zip(active, expected):
            torch.testing.assert_close(actual, reference)
        if not train_right:
            assert torch.equal(right, frozen_right)
            assert right.grad is None


@pytest.mark.parametrize("method", ["armijo", "wolfe"])
def test_linesearch_keeps_parameters_with_zero_gradients(method):
    """A computed zero gradient is distinct from a parameter absent from the objective."""
    active = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    zero_grad = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))
    optimizer = geoopt.optim.RiemannianLineSearch([active, zero_grad], line_search_method=method)

    def closure():
        optimizer.zero_grad()
        loss = active.square().sum() + 0 * zero_grad.sum()
        loss.backward()
        return loss.item()

    optimizer.step(closure)
    assert zero_grad.grad is not None
    assert torch.equal(zero_grad.grad, torch.zeros_like(zero_grad))
    assert zero_grad in optimizer.state
    torch.testing.assert_close(zero_grad, torch.tensor([7.0], dtype=torch.float64))
