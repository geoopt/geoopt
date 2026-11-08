import math

import pytest
import torch

from geoopt import Stereographic
from geoopt.manifolds.stereographic import math as smath


AXIS_CASES = [
    (batch_shape, dim)
    for batch_shape in [(), (4,), (2, 4), (2, 1, 4)]
    for dim in range(-len(batch_shape) - 1, len(batch_shape) + 1)
]


def reference_matvec(matrix, points, curvature):
    """Evaluate the scalar radial formula separately for each last-axis point."""
    results = []
    for point in points.reshape(-1, points.shape[-1]):
        product = matrix @ point
        if curvature == 0:
            results.append(product)
            continue
        radius = point.norm().item()
        product_radius = product.norm().item()
        root = math.sqrt(abs(curvature))
        inverse = math.atanh if curvature < 0 else math.atan
        forward = math.tanh if curvature < 0 else math.tan
        mapped_radius = forward(product_radius / radius * inverse(root * radius)) / root
        results.append(product * (mapped_radius / product_radius))
    return torch.stack(results).reshape(points.shape[:-1] + (matrix.shape[0],))


@pytest.mark.parametrize("batch_shape,dim", AXIS_CASES)
@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_matvec_preserves_coordinate_axis(batch_shape, dim, curvature, dtype, strided):
    points = torch.linspace(0.01, 0.24, math.prod(batch_shape) * 3, dtype=dtype)
    points = points.reshape(batch_shape + (3,))
    matrix = torch.linspace(-0.4, 0.8, 15, dtype=dtype).reshape(5, 3)
    x = points.movedim(-1, dim)
    if strided:
        x = torch.stack((x, x), dim=-1)[..., 0]
        matrix = torch.stack((matrix, matrix), dim=-1)[..., 0]
        assert not x.is_contiguous()
        assert not matrix.is_contiguous()
    expected = reference_matvec(matrix, points, curvature).movedim(-1, dim)
    result = smath.mobius_matvec(matrix, x, k=x.new_tensor(curvature), dim=dim)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dim", [0, -2])
@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("out_features", [1, 3])
def test_matvec_singleton_and_equal_batch_axes(dim, curvature, out_features):
    x = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.1, -0.1]], dtype=torch.float64)
    matrix = x.new_tensor([[1.0, 0.5], [-0.5, 1.0], [1.0, 1.0]])[:out_features]
    expected = reference_matvec(matrix, x.T, curvature).T
    result = smath.mobius_matvec(matrix, x, k=x.new_tensor(curvature), dim=dim)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dim", [0, 1, -3, -2])
@pytest.mark.parametrize("project", [False, True])
def test_matvec_wrapper_projects_on_coordinate_axis(dim, project):
    points = torch.tensor([[[0.1, 0.2], [0.2, -0.1], [0.3, 0.2]]], dtype=torch.float64)
    x = points.movedim(-1, dim)
    matrix = x.new_tensor([[30.0, 0.0], [0.0, 30.0], [20.0, 20.0], [-10.0, 10.0]])
    manifold = Stereographic(k=-1.0).to(dtype=x.dtype)
    expected = reference_matvec(matrix, points, -1.0)
    if project:
        expected = expected * ((1 - 1e-5) / expected.norm(dim=-1, keepdim=True)).clamp_max(1)
    expected = expected.movedim(-1, dim)
    result = manifold.mobius_matvec(matrix, x, dim=dim, project=project)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dim", [0, 1, -3, -2])
@pytest.mark.parametrize("curvature", [-0.5, 0.0, 0.5])
def test_matvec_axis_gradients(dim, curvature):
    x = torch.linspace(0.03, 0.18, 12, dtype=torch.float64).reshape(2, 3, 2)
    x = x.movedim(-1, dim).requires_grad_()
    matrix = x.new_tensor([[0.2, 0.1], [-0.1, 0.3], [0.4, -0.2]]).requires_grad_()
    k = x.new_tensor(curvature).requires_grad_()

    def fn(matrix, x, k):
        return smath.mobius_matvec(matrix, x, k=k, dim=dim)

    assert torch.autograd.gradcheck(fn, (matrix, x, k))
    weights = torch.linspace(0.1, 0.8, 18, dtype=x.dtype).reshape(2, 3, 3)

    def loss(matrix, x, k=k.detach()):
        return (fn(matrix, x, k).movedim(dim, -1) * weights).sum()

    def reference_loss(matrix, x, k=k.detach()):
        points = torch.stack(x.unbind(dim), dim=-1)
        if curvature == 0:
            return ((points @ matrix.T) * weights).sum()
        root = k.abs().sqrt()
        inverse = torch.atanh if curvature < 0 else torch.atan
        forward = torch.tanh if curvature < 0 else torch.tan
        radius = points.norm(dim=-1, keepdim=True)
        tangent = points * (inverse(root * radius) / (root * radius))
        product = tangent @ matrix.T
        product_radius = product.norm(dim=-1, keepdim=True)
        result = product * (forward(root * product_radius) / (root * product_radius))
        return (result * weights).sum()

    inputs = (matrix, x) if curvature == 0 else (matrix, x, k)
    actual = torch.autograd.functional.hessian(loss, inputs)
    expected = torch.autograd.functional.hessian(reference_loss, inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("dim", [0, -2])
@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
def test_matvec_zero_input_axis_jacobian(dim, curvature):
    x = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    matrix = x.new_tensor([[0.2, -0.1], [-0.4, 0.5], [0.1, 0.3], [0.4, 0.2]])
    k = x.new_tensor(curvature)

    def fn(x):
        return smath.mobius_matvec(matrix, x, k=k, dim=dim)

    torch.testing.assert_close(fn(x), torch.zeros(4, 3, dtype=x.dtype), atol=0, rtol=0)
    jacobian = torch.autograd.functional.jacobian(fn, x)
    expected = torch.einsum("oi,bj->obij", matrix, torch.eye(3, dtype=x.dtype))
    torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize("dim", [0, 1, 2, -3, -2, -1])
def test_matvec_broadcasted_curvature(dim):
    points = torch.linspace(0.01, 0.24, 24, dtype=torch.float64).reshape(2, 4, 3)
    curvatures = points.new_tensor([[-1.0, 0.0, 1.0, -0.5], [0.5, 1.0, 0.0, -1.0]])
    matrix = torch.linspace(-0.4, 0.8, 15, dtype=points.dtype).reshape(5, 3)
    expected = torch.stack(
        [
            reference_matvec(matrix, point, curvature.item())
            for point, curvature in zip(points.reshape(-1, 3), curvatures.reshape(-1))
        ]
    ).reshape(2, 4, 5)
    result = smath.mobius_matvec(
        matrix,
        points.movedim(-1, dim),
        k=curvatures.unsqueeze(-1).movedim(-1, dim),
        dim=dim,
    )
    torch.testing.assert_close(result, expected.movedim(-1, dim))


def test_matvec_batched_matrices_last_axis():
    x = torch.tensor([[0.1, 0.2], [0.2, -0.1]], dtype=torch.float64)
    matrices = torch.linspace(-0.2, 0.4, 12, dtype=x.dtype).reshape(2, 3, 2)
    expected = torch.stack([reference_matvec(m, p, -1.0) for m, p in zip(matrices, x)])
    torch.testing.assert_close(smath.mobius_matvec(matrices, x, k=x.new_tensor(-1.0)), expected)
    for dim in [0, 1, -2]:
        with pytest.raises((RuntimeError, torch.jit.Error), match="last dim only"):
            smath.mobius_matvec(matrices, x, k=x.new_tensor(-1.0), dim=dim)
