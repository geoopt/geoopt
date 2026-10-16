import pytest
import torch

from geoopt.manifolds.lorentz import math


@pytest.mark.parametrize("shape,dim", [((4,), -1), ((2, 3, 4), -1), ((4, 2, 3), 0),
                                      ((2, 4, 3), 1), ((2, 4, 3), -2)])
@pytest.mark.parametrize("curvature", [0.5, 2.0])
def test_lorentz_to_poincare_coordinate_dimension(shape, dim, curvature):
    k = torch.tensor(curvature, dtype=torch.float64)
    generator = torch.Generator().manual_seed(17)
    ambient = torch.randn(shape, generator=generator, dtype=torch.float64)
    x = math.project(ambient, k=k, dim=dim).requires_grad_()
    result = math.lorentz_to_poincare(x, k=k, dim=dim)
    moved = x.movedim(dim, -1)
    expected = (moved[..., 1:] / (moved[..., :1] + k.sqrt())).movedim(-1, dim)
    torch.testing.assert_close(result, expected)
    recovered = math.poincare_to_lorentz(result, k=k, dim=dim, eps=0)
    torch.testing.assert_close(recovered, x)
    gradient = torch.autograd.grad(result.sum(), x)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(gradient, expected_gradient)
