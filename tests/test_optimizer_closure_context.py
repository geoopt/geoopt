import pytest
import torch
import torch.nn.functional as F

import geoopt


@pytest.mark.parametrize("outer_grad_enabled", [False, True])
@pytest.mark.parametrize(
    "optimizer_type,reference_type,sparse",
    [
        (geoopt.optim.RiemannianAdam, torch.optim.Adam, False),
        (geoopt.optim.RiemannianSGD, torch.optim.SGD, False),
        (geoopt.optim.SparseRiemannianAdam, torch.optim.SparseAdam, True),
        (geoopt.optim.SparseRiemannianSGD, torch.optim.SGD, True),
    ],
)
def test_optimizer_closure_grad_context(optimizer_type, reference_type, sparse, outer_grad_enabled):
    point = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    )
    reference = torch.nn.Parameter(point.detach().clone())
    options = {"lr": 0.1}
    if reference_type in (torch.optim.Adam, torch.optim.SparseAdam):
        options["eps"] = 1e-15
    optimizer = optimizer_type([point], **options)
    reference_optimizer = reference_type([reference], **options)
    indices = torch.tensor([0, 2])
    closure_losses = []
    modes = []

    def closure():
        modes.append(torch.is_grad_enabled())
        optimizer.zero_grad()
        values = F.embedding(indices, point, sparse=True) if sparse else point
        loss = values.square().sum()
        loss.backward()
        closure_losses.append(loss)
        return loss

    def reference_closure():
        reference_optimizer.zero_grad()
        values = F.embedding(indices, reference, sparse=True) if sparse else reference
        loss = values.square().sum()
        loss.backward()
        return loss

    for _ in range(2):
        with torch.set_grad_enabled(outer_grad_enabled):
            result = optimizer.step(closure)
            expected_loss = reference_optimizer.step(reference_closure)
            assert torch.is_grad_enabled() == outer_grad_enabled
        assert result is closure_losses[-1]
        assert result.requires_grad
        torch.testing.assert_close(result, expected_loss)
        torch.testing.assert_close(point, reference, rtol=1e-12, atol=1e-12)
    assert modes == [True, True]


@pytest.mark.parametrize("outer_grad_enabled", [False, True])
@pytest.mark.parametrize("method", ["armijo", "wolfe"])
@pytest.mark.parametrize("maximize", [False, True])
def test_line_search_closure_grad_context(method, outer_grad_enabled, maximize):
    point = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method=method, cg_method="steepest", maximize=maximize
    )
    modes = []

    def closure():
        modes.append(torch.is_grad_enabled())
        optimizer.zero_grad()
        loss = (-1 if maximize else 1) * point.square().sum()
        loss.backward()
        return loss.item()

    with torch.set_grad_enabled(outer_grad_enabled):
        result = optimizer.step(closure)
        assert torch.is_grad_enabled() == outer_grad_enabled
    assert len(modes) >= 2
    assert all(modes)
    assert abs(result) < 1e-12
    torch.testing.assert_close(point, torch.zeros_like(point), rtol=0.0, atol=1e-12)
