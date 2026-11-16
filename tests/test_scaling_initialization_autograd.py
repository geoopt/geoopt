import math

import geoopt
import pytest
import torch


@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("kind", ["tensor", "parameter", "expression", "dtype_conversion"])
def test_scaled_initialization_preserves_caller_autograd(kind, learnable):
    if kind == "parameter":
        source = torch.nn.Parameter(torch.tensor(2.0))
    elif kind == "expression":
        source = torch.tensor(math.log(2.0), requires_grad=True)
    elif kind == "dtype_conversion":
        source = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    else:
        source = torch.tensor(2.0, requires_grad=True)
    scale = source.exp() if kind == "expression" else source
    source_value = source.detach().clone()
    scaled = geoopt.Scaled(geoopt.Euclidean(ndim=1), scale, learnable=learnable)
    assert source.requires_grad
    assert scale.requires_grad
    torch.testing.assert_close(source, source_value)
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    distance = scaled.dist(x, torch.zeros_like(x))
    torch.testing.assert_close(distance, torch.tensor(10.0))
    (distance + source.square()).backward()
    torch.testing.assert_close(source.grad, 2 * source_value)
    torch.testing.assert_close(x.grad, torch.tensor([1.2, 1.6]))
    if learnable:
        assert isinstance(scaled._log_scale, torch.nn.Parameter)
        torch.testing.assert_close(scaled._log_scale.grad, distance.detach())
        assert scaled._log_scale is not source
    else:
        assert not scaled._scale.requires_grad
        assert scaled._scale.grad_fn is None


@pytest.mark.parametrize("learnable", [False, True])
def test_nested_scaled_initialization_preserves_inner_training(learnable):
    inner = geoopt.Scaled(geoopt.Euclidean(ndim=1), 2.0, learnable=True)
    outer = geoopt.Scaled(inner, inner.scale, learnable=learnable)
    x = torch.tensor([3.0, 4.0])
    distance = outer.dist(x, torch.zeros_like(x))
    torch.testing.assert_close(distance, torch.tensor(20.0))
    distance.backward()
    torch.testing.assert_close(inner._log_scale.grad, torch.tensor(20.0))
    if learnable:
        torch.testing.assert_close(outer._log_scale.grad, torch.tensor(20.0))


@pytest.mark.parametrize("learnable", [False, True])
@pytest.mark.parametrize("scale", [2, 2.0, torch.tensor(2.0)])
def test_scaled_constant_initialization_and_state_round_trip(scale, learnable):
    scaled = geoopt.Scaled(geoopt.Euclidean(ndim=1), scale, learnable=learnable)
    restored = geoopt.Scaled(geoopt.Euclidean(ndim=1), 1.0, learnable=learnable)
    restored.load_state_dict(scaled.state_dict())
    x = torch.tensor([3.0, 4.0])
    torch.testing.assert_close(restored.dist(x, torch.zeros_like(x)), torch.tensor(10.0))
    assert restored.scale.requires_grad == learnable
