import pytest
import torch

from geoopt.manifolds.lorentz import math


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0, 9.0])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_polar_projection_respects_curvature_and_radial_distance(dtype, curvature, dim):
    directions = torch.tensor(
        [[[1.0, 2.0, -1.0], [2.0, -1.0, 2.0], [1.0, 0.0, 0.0]],
         [[1.0, -1.0, 2.0], [0.0, 1.0, 0.0], [2.0, 2.0, -1.0]]], dtype=dtype
    )
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    radius = torch.tensor([[0.0, 0.2, 0.7], [-0.4, 1.0, 1.4]], dtype=dtype)
    k = torch.tensor(curvature, dtype=dtype)
    polar = torch.cat((directions, radius.unsqueeze(-1)), dim=-1).movedim(-1, dim)
    result = math.project_polar(polar, k=k, dim=dim)
    atol = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        math.inner(result, result, dim=dim), torch.full_like(radius, -curvature),
        atol=atol, rtol=atol,
    )
    canonical = result.movedim(dim, -1)
    expected_origin = torch.cat((k.sqrt().reshape(1), torch.zeros(3, dtype=dtype)))
    torch.testing.assert_close(canonical[0, 0], expected_origin, atol=0, rtol=0)
    tangent = torch.cat((torch.zeros_like(radius.unsqueeze(-1)), radius.unsqueeze(-1) * directions), dim=-1)
    expected = math.expmap0(tangent, k=k)
    nonzero = radius != 0
    torch.testing.assert_close(canonical[nonzero], expected[nonzero], atol=atol, rtol=atol)
    torch.testing.assert_close(
        math.dist0(canonical[nonzero], k=k), radius[nonzero].abs(), atol=atol, rtol=atol
    )


@pytest.mark.parametrize("curvature", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("dim", [-1, 0])
def test_polar_radius_and_curvature_gradients_match_exponential_map(curvature, dim):
    direction = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    direction = direction / direction.norm()
    radius = torch.tensor([[0.2, 0.8], [-0.3, 1.0]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor(curvature, dtype=torch.float64, requires_grad=True)
    directions = direction.expand(radius.shape + (3,))
    polar = torch.cat((directions, radius.unsqueeze(-1)), dim=-1).movedim(-1, dim)
    actual = math.project_polar(polar, k=k, dim=dim).movedim(dim, -1)
    tangent = torch.cat((torch.zeros_like(radius.unsqueeze(-1)), radius.unsqueeze(-1) * directions), dim=-1)
    expected = math.expmap0(tangent, k=k)
    weights = torch.tensor([0.7, -1.3, 0.4, 2.0], dtype=torch.float64)
    actual_grads = torch.autograd.grad((actual * weights).sum(), (radius, k), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (radius, k))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)
