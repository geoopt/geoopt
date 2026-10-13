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
