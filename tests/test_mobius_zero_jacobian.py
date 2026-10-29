import math
import pytest
import torch
from geoopt.manifolds.stereographic import math as smath


def gain(x, curvature):
    norm = x.norm().item()
    if curvature < 0:
        return math.atanh(math.sqrt(-curvature) * norm) / (math.sqrt(-curvature) * norm)
    if curvature > 0:
        return math.atan(math.sqrt(curvature) * norm) / (math.sqrt(curvature) * norm)
    return 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("kind", ["matrix", "pointwise"])
def test_zero_weight_jacobian(dtype, curvature, kind):
    x = torch.tensor([0.2, -0.3, 0.1], dtype=dtype)
    k = x.new_tensor(curvature)
    weight = torch.zeros((2, 3) if kind == "matrix" else (3,), dtype=dtype, requires_grad=True)
    fn = (
        (lambda w: smath.mobius_matvec(w, x, k=k))
        if kind == "matrix"
        else (lambda w: smath.mobius_pointwise_mul(w, x, k=k))
    )
    jac = torch.autograd.functional.jacobian(fn, weight)
    expected = (
        torch.eye(2, dtype=dtype)[:, :, None] * (gain(x, curvature) * x)[None, None, :]
        if kind == "matrix"
        else torch.diag(gain(x, curvature) * x)
    )
    torch.testing.assert_close(jac, expected)
    # A centered difference checks the derivative independently of the implementation's autograd graph.
    direction = torch.ones_like(weight)
    eps = 1e-3 if dtype == torch.float32 else 1e-5
    numerical = (fn(weight + eps * direction) - fn(weight - eps * direction)) / (2 * eps)
    analytic = torch.einsum("ijk,jk->i", jac, direction) if kind == "matrix" else jac @ direction
    torch.testing.assert_close(analytic, numerical, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["matrix", "pointwise"])
def test_zero_initialized_weight_can_learn(kind):
    x = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
    weight = torch.nn.Parameter(torch.zeros((2, 3) if kind == "matrix" else (3,), dtype=x.dtype))
    optimizer = torch.optim.SGD([weight], lr=0.01)
    fn = (
        (lambda: smath.mobius_matvec(weight, x, k=x.new_tensor(-1.0)))
        if kind == "matrix"
        else (lambda: smath.mobius_pointwise_mul(weight, x, k=x.new_tensor(-1.0)))
    )
    initial = (fn() - 1).square().sum()
    initial.backward()
    optimizer.step()
    assert torch.count_nonzero(weight).item() > 0
    assert (fn() - 1).square().sum() < initial


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
def test_small_pointwise_outputs_are_not_zeroed(curvature):
    x = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
    w = torch.full_like(x, 1e-10)
    k = x.new_tensor(curvature)
    result = smath.mobius_pointwise_mul(w, x, k=k)
    expected = smath.expmap0(w * smath.logmap0(x, k=k), k=k)
    torch.testing.assert_close(result, expected, atol=0, rtol=1e-12)


@pytest.mark.parametrize("curvature", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("kind", ["matrix", "pointwise"])
def test_zero_input_jacobian(curvature, kind):
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    k = x.new_tensor(curvature)
    if kind == "matrix":
        weight = x.new_tensor([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
        expected = weight
    else:
        weight = x.new_tensor([0.2, -0.1, 0.3])
        expected = torch.diag(weight)

    def fn(a):
        if kind == "matrix":
            return smath.mobius_matvec(weight, a, k=k)
        return smath.mobius_pointwise_mul(weight, a, k=k)

    torch.testing.assert_close(fn(x), torch.zeros(expected.shape[0], dtype=x.dtype), atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.functional.jacobian(fn, x), expected)


def test_nonzero_matrix_with_input_in_nullspace():
    # A trained/nonzero matrix can also produce zero, not only an initial zero weight.
    x = torch.tensor([0.1875, -0.09375, 0.03125], dtype=torch.float64, requires_grad=True)
    weight = x.new_tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    k = x.new_tensor(-1.0)
    assert torch.count_nonzero(weight @ x).item() == 0
    result = smath.mobius_matvec(weight, x, k=k)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
    jac = torch.autograd.functional.jacobian(lambda a: smath.mobius_matvec(weight, a, k=k), x)
    torch.testing.assert_close(jac, gain(x, -1.0) * weight)
