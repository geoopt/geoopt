import pytest
import torch

import geoopt



import math


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("manifold_type", [geoopt.Sphere, geoopt.SphereExact])
def test_zero_tangent_step_has_finite_analytic_jacobian(dtype, manifold_type):
    sphere = manifold_type()
    x = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    u = torch.zeros(3, dtype=dtype, requires_grad=True)
    actual = sphere.expmap(x, u)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    jacobian = torch.autograd.functional.jacobian(lambda tangent: sphere.expmap(x, tangent), u)
    expected = torch.diag(torch.tensor([0.0, 1.0, 1.0], dtype=dtype))
    assert torch.isfinite(jacobian).all()
    torch.testing.assert_close(jacobian, expected)


def test_mixed_zero_and_nonzero_sphere_steps_preserve_gradients():
    sphere = geoopt.SphereExact()
    angles = [0.0, 1e-10, 0.2, -0.4]
    x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(4, -1)
    u = torch.tensor([[0.0, angle, 0.0] for angle in angles], dtype=torch.float64, requires_grad=True)
    actual = sphere.expmap(x, u)
    expected = torch.tensor([[math.cos(angle), math.sin(angle), 0.0] for angle in angles], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    gradient = torch.autograd.grad(actual[:, 1].sum(), u)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[:, 1], torch.tensor([math.cos(angle) for angle in angles], dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)


def test_sphere_expmap_gradcheck_at_zero_tangent():
    sphere = geoopt.SphereExact()
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    coordinates = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    # Numerical perturbations must stay in the sphere's two-dimensional tangent
    # space; expmap is only defined for tangent vectors at its base point.
    def expmap_coordinates(tangent_coordinates):
        tangent = torch.cat((torch.zeros(1, dtype=x.dtype), tangent_coordinates))
        return sphere.expmap(x, tangent)
    assert torch.autograd.gradcheck(expmap_coordinates, (coordinates,))
