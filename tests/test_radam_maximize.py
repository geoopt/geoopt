import math

import pytest
import torch

import geoopt


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.15])
@pytest.mark.parametrize("amsgrad", [False, True])
def test_radam_matches_adam_maximize(maximize, weight_decay, amsgrad):
    point = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    reference = torch.nn.Parameter(point.detach().clone())
    options = dict(lr=0.1, maximize=maximize, weight_decay=weight_decay, amsgrad=amsgrad)
    optimizer = geoopt.optim.RiemannianAdam([point], **options)
    expected_optimizer = torch.optim.Adam([reference], **options)
    goal = torch.tensor([1.0, -0.8], dtype=torch.float64)
    for _ in range(3):
        point.grad = 2 * (goal - point.detach())
        reference.grad = 2 * (goal - reference.detach())
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(point, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("maximize", [False, True])
def test_radam_maximize_on_sphere(maximize):
    point = geoopt.ManifoldParameter(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), manifold=geoopt.Sphere()
    )
    optimizer = geoopt.optim.RiemannianAdam([point], lr=0.1, eps=0.0, maximize=maximize)
    point[0].backward()
    optimizer.step()
    sign = 1 if maximize else -1
    expected = torch.tensor([sign * 0.1, 1.0, 0.0], dtype=torch.float64) / math.sqrt(1.01)
    torch.testing.assert_close(point.detach(), expected, rtol=1e-12, atol=1e-12)


def test_radam_maximize_parameter_group_override():
    ascent = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    descent = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    optimizer = geoopt.optim.RiemannianAdam(
        [{"params": [ascent], "maximize": True}, {"params": [descent]}],
        lr=0.1,
        eps=0.0,
        maximize=False,
    )
    ascent.grad = torch.ones_like(ascent)
    descent.grad = torch.ones_like(descent)
    optimizer.step()
    torch.testing.assert_close(ascent, torch.tensor([0.1], dtype=torch.float64))
    torch.testing.assert_close(descent, torch.tensor([-0.1], dtype=torch.float64))


@pytest.mark.parametrize(
    "optimizer_type,momentum",
    [
        (geoopt.optim.RiemannianSGD, 0.0),
        (geoopt.optim.RiemannianSGD, 0.6),
        (geoopt.optim.SparseRiemannianSGD, 0.0),
        (geoopt.optim.SparseRiemannianSGD, 0.6),
        (geoopt.optim.SparseRiemannianAdam, 0.0),
    ],
)
@pytest.mark.parametrize("maximize", [False, True])
def test_other_optimizers_maximize(optimizer_type, maximize, momentum):
    """Maximization equals minimization of the negative objective over several steps."""
    sparse = optimizer_type is not geoopt.optim.RiemannianSGD
    point = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64))
    reference = torch.nn.Parameter(point.detach().clone())
    options = dict(lr=0.1)
    if optimizer_type is not geoopt.optim.SparseRiemannianAdam:
        options.update(momentum=momentum, nesterov=momentum > 0)
    optimizer = optimizer_type([point], maximize=maximize, **options)
    expected_optimizer = optimizer_type([reference], **options)
    for step in range(3):
        gradient = torch.tensor([[0.3 + step, -0.7]], dtype=torch.float64)
        if sparse:
            grad = torch.sparse_coo_tensor(torch.tensor([[step % 2]]), gradient, point.shape)
        else:
            grad = gradient.expand_as(point).clone()
        point.grad = grad.clone()
        reference.grad = -grad if maximize else grad.clone()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(point, reference, rtol=1e-12, atol=1e-12)
        if sparse:
            torch.testing.assert_close(point.grad.to_dense(), grad.to_dense())
        elif maximize:
            torch.testing.assert_close(point.grad, grad)


@pytest.mark.parametrize(
    "optimizer_type",
    [
        geoopt.optim.RiemannianSGD,
        geoopt.optim.SparseRiemannianSGD,
        geoopt.optim.SparseRiemannianAdam,
    ],
)
def test_other_optimizers_maximize_group_override(optimizer_type):
    """An explicit group override takes precedence over the constructor default."""
    ascent = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    descent = torch.nn.Parameter(torch.zeros_like(ascent))
    optimizer = optimizer_type(
        [{"params": [ascent]}, {"params": [descent], "maximize": False}], lr=0.1, maximize=True
    )
    grad = torch.ones_like(ascent)
    if optimizer_type is not geoopt.optim.RiemannianSGD:
        grad = grad.to_sparse(sparse_dim=1)
    ascent.grad = grad.clone()
    descent.grad = grad.clone()
    optimizer.step()
    torch.testing.assert_close(ascent, torch.full_like(ascent, 0.1))
    torch.testing.assert_close(descent, torch.full_like(descent, -0.1))


@pytest.mark.parametrize("method", ["armijo", "wolfe"])
@pytest.mark.parametrize("cg_method", ["steepest", "fr", "pr"])
@pytest.mark.parametrize("maximize", [False, True])
def test_linesearch_maximize(method, cg_method, maximize):
    """Line search signs both its scalar objective and derivative, returning the original objective."""
    point = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    reference = torch.nn.Parameter(point.detach().clone())
    options = dict(line_search_method=method, cg_method=cg_method)
    optimizer = geoopt.optim.RiemannianLineSearch([point], maximize=maximize, **options)
    expected_optimizer = geoopt.optim.RiemannianLineSearch([reference], **options)
    sign = -1 if maximize else 1

    def closure():
        optimizer.zero_grad()
        loss = sign * ((point - 2) ** 2).sum()
        loss.backward()
        return loss.item()

    def reference_closure():
        expected_optimizer.zero_grad()
        loss = ((reference - 2) ** 2).sum()
        loss.backward()
        return loss.item()

    for _ in range(2):
        actual = optimizer.step(closure)
        expected = expected_optimizer.step(reference_closure)
        torch.testing.assert_close(point, reference, rtol=1e-12, atol=1e-12)
        assert actual == pytest.approx(sign * expected)
        assert optimizer.last_step_size == pytest.approx(expected_optimizer.last_step_size)


def test_linesearch_maximize_consistent_groups():
    """A single line search cannot minimize and maximize the same scalar objective."""
    x = torch.nn.Parameter(torch.zeros(1))
    y = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="maximize"):
        geoopt.optim.RiemannianLineSearch(
            [{"params": [x], "maximize": True}, {"params": [y], "maximize": False}]
        )


@pytest.mark.parametrize(
    "optimizer_type,reference_type,sparse",
    [
        (geoopt.optim.RiemannianSGD, torch.optim.SGD, False),
        (geoopt.optim.SparseRiemannianSGD, torch.optim.SGD, True),
        (geoopt.optim.SparseRiemannianAdam, torch.optim.SparseAdam, True),
    ],
)
@pytest.mark.parametrize("maximize", [False, True])
def test_other_optimizers_maximize_native_reference(
    optimizer_type, reference_type, sparse, maximize
):
    """Euclidean fixed-step updates agree with the native PyTorch optimizers."""
    point = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64))
    reference = torch.nn.Parameter(point.detach().clone())
    options = {"lr": 0.1, "maximize": maximize}
    if optimizer_type is geoopt.optim.RiemannianSGD:
        options["weight_decay"] = 0.15
    if optimizer_type is geoopt.optim.SparseRiemannianAdam:
        # Isolate direction from the existing difference in epsilon placement.
        options["eps"] = 1e-15
    optimizer = optimizer_type([point], **options)
    expected_optimizer = reference_type([reference], **options)
    for step in range(3):
        grad = torch.full_like(point, 0.5 + step)
        if sparse:
            grad = grad.to_sparse(sparse_dim=1)
        point.grad = grad.clone()
        reference.grad = grad.clone()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(point, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "optimizer_type",
    [
        geoopt.optim.RiemannianAdam,
        geoopt.optim.RiemannianSGD,
        geoopt.optim.SparseRiemannianAdam,
        geoopt.optim.SparseRiemannianSGD,
    ],
)
def test_maximize_missing_flag_retains_minimization(optimizer_type):
    """Old parameter groups without the new key retain minimization behavior."""
    point = torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.float64))
    optimizer = optimizer_type([point], lr=0.1)
    optimizer.param_groups[0].pop("maximize", None)
    grad = torch.ones_like(point)
    if optimizer_type in (geoopt.optim.SparseRiemannianAdam, geoopt.optim.SparseRiemannianSGD):
        grad = grad.to_sparse(sparse_dim=1)
    point.grad = grad
    optimizer.step()
    torch.testing.assert_close(point, torch.full_like(point, -0.1))
