import pytest
import torch

import geoopt



@pytest.mark.parametrize("options", [
    {"momentum": 0.9}, {"momentum": 0.9, "nesterov": True},
    {"momentum": 0.9, "dampening": 0.2},
    {"momentum": 0.9, "weight_decay": 0.3},
    {"momentum": 0.0}])
def test_euclidean_momentum_matches_torch_sgd_each_step(options):
    actual = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
    expected = torch.nn.Parameter(actual.detach().clone())
    optimizer = geoopt.optim.RiemannianSGD([actual], lr=0.1, **options)
    reference = torch.optim.SGD([expected], lr=0.1, **options)
    for step in range(5):
        gradient = torch.tensor([0.3 + step * 0.1, -0.4, 0.2 - step * 0.03], dtype=torch.float64)
        actual.grad = gradient.clone()
        expected.grad = gradient.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        if options["momentum"]:
            torch.testing.assert_close(optimizer.state[actual]["momentum_buffer"],
                                       reference.state[expected]["momentum_buffer"], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("nesterov", [False, True])
def test_first_sphere_momentum_is_the_riemannian_gradient(nesterov):
    sphere = geoopt.Sphere()
    point = geoopt.ManifoldParameter(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), manifold=sphere)
    point.grad = torch.tensor([2.0, 0.3, -0.4], dtype=torch.float64)
    gradient = torch.tensor([0.0, 0.3, -0.4], dtype=torch.float64)
    momentum, lr = 0.9, 0.1
    direction = gradient * (1 + momentum if nesterov else 1)
    raw_point = point.detach() - lr * direction
    expected = raw_point / torch.linalg.vector_norm(raw_point)
    expected_buffer = gradient - torch.dot(expected, gradient) * expected
    optimizer = geoopt.optim.RiemannianSGD([point], lr=lr, momentum=momentum, nesterov=nesterov)
    optimizer.step()
    torch.testing.assert_close(point, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(optimizer.state[point]["momentum_buffer"], expected_buffer,
                               rtol=1e-13, atol=1e-13)


def test_late_gradient_initializes_momentum_once():
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = geoopt.optim.RiemannianSGD([first, second], lr=0.1, momentum=0.9)
    first.grad = torch.ones_like(first)
    optimizer.step()
    assert not optimizer.state[second]
    second.grad = torch.ones_like(second)
    optimizer.step()
    torch.testing.assert_close(second, torch.tensor([1.9], dtype=torch.float64), rtol=1e-13, atol=1e-13)
