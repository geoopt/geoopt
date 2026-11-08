import math

import pytest
import torch

import geoopt


@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 1, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("counts", [(1, 1), (25, 25), (26, 2), (2, 26)])
def test_cdist_coincident_points_have_zero_gradients(shape, dtype, counts):
    manifold = geoopt.Euclidean(ndim=len(shape))
    x = torch.full((counts[0],) + shape, 3, dtype=dtype, requires_grad=True)
    y = torch.full((counts[1],) + shape, 3, dtype=dtype, requires_grad=True)

    distances = manifold.cdist(x, y)
    torch.testing.assert_close(distances, distances.new_zeros(counts))
    for actual, point in zip(torch.autograd.grad(distances.sum(), (x, y)), (x, y)):
        torch.testing.assert_close(actual, torch.zeros_like(point))


@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 1, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cdist_self_distance_loss_has_finite_gradient(shape, dtype):
    manifold = geoopt.Euclidean(ndim=len(shape))
    points = torch.tensor([[0, 0], [3, 4], [3, 4]], dtype=dtype)
    points = points.reshape((3,) + shape).requires_grad_()

    distances = manifold.cdist(points, points)
    torch.testing.assert_close(distances, points.new_tensor([[0, 5, 5], [5, 0, 0], [5, 0, 0]]))
    gradient = torch.autograd.grad(distances.sum(), points)[0]
    expected = points.new_tensor([[-2.4, -3.2], [1.2, 1.6], [1.2, 1.6]])
    torch.testing.assert_close(gradient, expected.reshape_as(points))
    updated = manifold.retr(points.detach(), -0.1 * gradient)
    assert torch.isfinite(updated).all()
    assert manifold.cdist(updated, updated).sum() < distances.sum()


@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 1, 2)])
def test_cdist_broadcast_noncontiguous_values_and_gradients(shape):
    manifold = geoopt.Euclidean(ndim=len(shape))
    x = torch.arange(24, dtype=torch.float64).reshape(2, 1, 3, 4)[..., ::2]
    y = torch.arange(16, dtype=torch.float64).reshape(1, 4, 1, 4)[..., ::2]
    x = x.reshape((2, 1, 3) + shape).requires_grad_()
    y = y.reshape((1, 4, 1) + shape).requires_grad_()
    assert not x.is_contiguous()
    assert not y.is_contiguous()

    distances = manifold.cdist(x, y)
    difference = x.unsqueeze(-len(shape) - 1) - y.unsqueeze(-len(shape) - 2)
    expected = torch.linalg.vector_norm(difference, dim=tuple(range(-len(shape), 0)))
    assert distances.shape == (2, 4, 3, 1)
    torch.testing.assert_close(distances, expected)
    actual_gradient = torch.autograd.grad(distances.sum(), (x, y))
    expected_gradient = torch.autograd.grad(expected.sum(), (x, y))
    for actual, reference in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 1, 2)])
def test_cdist_nonzero_first_and_second_derivatives(shape):
    manifold = geoopt.Euclidean(ndim=len(shape))
    x = torch.tensor([[0.25, 0.5], [1, 2]], dtype=torch.float64)
    y = torch.tensor([[3, 4], [-2, -3]], dtype=torch.float64)
    x = x.reshape((2,) + shape).requires_grad_()
    y = y.reshape((2,) + shape).requires_grad_()

    expected = torch.tensor(
        [[math.dist(a, b) for b in y.flatten(1).tolist()] for a in x.flatten(1).tolist()],
        dtype=torch.float64,
    )
    torch.testing.assert_close(manifold.cdist(x, y), expected)
    assert torch.autograd.gradcheck(manifold.cdist, (x, y))
    assert torch.autograd.gradgradcheck(manifold.cdist, (x, y))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cdist_preserves_floating_dtype_and_zero_gradients(ndim, dtype):
    manifold = geoopt.Euclidean(ndim=ndim)
    shape = (1,) * ndim
    x = torch.tensor([0, 3], dtype=dtype).reshape((2,) + shape).requires_grad_()
    y = torch.tensor([0, 4], dtype=dtype).reshape((2,) + shape).requires_grad_()
    try:
        x.new_zeros(()).clamp(min=0)
    except RuntimeError as error:
        assert dtype == torch.float16
        assert "not implemented for 'Half'" in str(error)
        with pytest.raises(RuntimeError, match=str(error)):
            manifold.cdist(x, y)
        return
    distances = manifold.cdist(x, y)
    assert distances.dtype == dtype
    torch.testing.assert_close(distances, x.new_tensor([[0, 4], [3, 1]]))
    gx, gy = torch.autograd.grad(distances.sum(), (x, y))
    torch.testing.assert_close(gx, x.new_tensor([-1, 0]).reshape_as(x))
    torch.testing.assert_close(gy, y.new_tensor([-1, 2]).reshape_as(y))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cdist_integer_inputs(ndim, dtype):
    shape = (1,) * (ndim - 1) + (2,)
    x = torch.tensor([[0, 0], [3, 4]], dtype=dtype).reshape((2,) + shape)
    distances = geoopt.Euclidean(ndim=ndim).cdist(x, x)
    assert distances.dtype == torch.get_default_dtype()
    torch.testing.assert_close(distances, torch.tensor([[0.0, 5.0], [5.0, 0.0]]))


@pytest.mark.parametrize("shape", [(0,), (0, 2), (2, 0, 2)])
def test_cdist_empty_manifold_dimensions(shape):
    x = torch.empty((3,) + shape, dtype=torch.float64, requires_grad=True)
    y = torch.empty((4,) + shape, dtype=torch.float64, requires_grad=True)
    distances = geoopt.Euclidean(ndim=len(shape)).cdist(x, y)
    torch.testing.assert_close(distances, torch.zeros(3, 4, dtype=torch.float64))
    for gradient, point in zip(torch.autograd.grad(distances.sum(), (x, y)), (x, y)):
        torch.testing.assert_close(gradient, torch.zeros_like(point))


@pytest.mark.parametrize("counts", [(0, 3), (3, 0), (0, 0)])
def test_cdist_empty_point_batches(counts):
    x = torch.empty(counts[0], 2, requires_grad=True)
    y = torch.empty(counts[1], 2, requires_grad=True)
    distances = geoopt.Euclidean(ndim=1).cdist(x, y)
    assert distances.shape == counts
    for gradient, point in zip(torch.autograd.grad(distances.sum(), (x, y)), (x, y)):
        torch.testing.assert_close(gradient, torch.zeros_like(point))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cdist_nonfinite_inputs_are_not_masked_to_zero(ndim, value):
    shape = (1,) * (ndim - 1) + (2,)
    x = torch.tensor([[value, 0]]).reshape((1,) + shape)
    y = torch.zeros_like(x)
    distances = geoopt.Euclidean(ndim=ndim).cdist(x, y)
    assert torch.isnan(distances).all()


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_cdist_infinite_positive_radicand_is_preserved(ndim):
    shape = (1,) * (ndim - 1) + (2,)
    x = torch.tensor([[1e20, 0]]).reshape((1,) + shape)
    y = torch.zeros_like(x)
    distances = geoopt.Euclidean(ndim=ndim).cdist(x, y)
    assert torch.isposinf(distances).all()


def test_cdist_scalar_manifold_is_unchanged():
    x = torch.tensor([0.0, 3.0], requires_grad=True)
    y = torch.tensor([0.0, 4.0], requires_grad=True)
    distances = geoopt.Euclidean(ndim=0).cdist(x, y)
    torch.testing.assert_close(distances, torch.tensor([[0.0, 4.0], [3.0, 1.0]]))
    gx, gy = torch.autograd.grad(distances.sum(), (x, y))
    torch.testing.assert_close(gx, torch.tensor([-1.0, 0.0]))
    torch.testing.assert_close(gy, torch.tensor([-1.0, 2.0]))
