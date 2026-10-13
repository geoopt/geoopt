import geoopt
import pytest
import torch


@pytest.mark.parametrize("tensor_type", [geoopt.ManifoldTensor, geoopt.ManifoldParameter])
@pytest.mark.parametrize("p", [0, 1, 2, 3, float("inf"), -float("inf")])
@pytest.mark.parametrize("broadcast", [False, True])
def test_dist_matches_torch_norm_and_gradient(tensor_type, p, broadcast):
    data = torch.tensor([[1.0, 3.0], [6.0, 10.0]], dtype=torch.float64)
    other = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
    if not broadcast:
        other = other.expand_as(data)
    point = tensor_type(data.clone(), requires_grad=True)
    reference = data.clone().requires_grad_()

    actual = point.dist(other, p=p)
    expected = reference.dist(other, p=p)

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, point, allow_unused=True)[0]
    expected_gradient = torch.autograd.grad(expected, reference, allow_unused=True)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("tensor_type", [geoopt.ManifoldTensor, geoopt.ManifoldParameter])
@pytest.mark.parametrize("keepdim", [False, True])
def test_dist_geodesic_preserves_manifold_kwargs(tensor_type, keepdim):
    manifold = geoopt.Sphere()
    point = tensor_type(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64), manifold=manifold)
    other = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

    actual = point.dist(other, p="g", keepdim=keepdim)
    expected = manifold.dist(point, other, keepdim=keepdim)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
