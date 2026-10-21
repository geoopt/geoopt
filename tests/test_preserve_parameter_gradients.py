import pytest
import torch

import geoopt



@pytest.mark.parametrize("optimizer_type", [geoopt.optim.RiemannianSGD, geoopt.optim.RiemannianAdam])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weight_decay_preserves_stored_gradients_and_shared_views(optimizer_type, dtype):
    point = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=dtype))
    storage = torch.tensor([0.3, 9.0, -0.4, 9.0, 0.2, 9.0], dtype=dtype)
    gradient = storage[::2]
    point.grad = gradient
    saved = storage.clone()
    optimizer = optimizer_type([point], lr=0.1, weight_decay=0.3)
    optimizer.step()
    assert point.grad is gradient
    torch.testing.assert_close(storage, saved, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_type,reference_type", [
    (geoopt.optim.RiemannianSGD, torch.optim.SGD),
    (geoopt.optim.RiemannianAdam, torch.optim.Adam)])
def test_reused_gradient_does_not_accumulate_optimizer_regularization(optimizer_type, reference_type):
    point = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
    expected = torch.nn.Parameter(point.detach().clone())
    gradient = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)
    point.grad = gradient.clone()
    expected.grad = gradient.clone()
    optimizer = optimizer_type([point], lr=0.1, weight_decay=0.3)
    reference = reference_type([expected], lr=0.1, weight_decay=0.3)
    for _ in range(4):
        optimizer.step()
        reference.step()
        torch.testing.assert_close(point, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(point.grad, gradient, rtol=0, atol=0)


def test_nesterov_does_not_rewrite_the_backpropagated_gradient():
    point = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = geoopt.optim.RiemannianSGD([point], lr=0.1, momentum=0.9, nesterov=True)
    point.square().sum().backward()
    gradient = point.grad.clone()
    optimizer.step()
    torch.testing.assert_close(point.grad, gradient, rtol=0, atol=0)
