import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (4,), (2, 3), (0,)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_diagonal_metric_and_retraction(dtype, batch, keepdim):
    manifold = geoopt.SymmetricPositiveDefinite()
    diagonal = torch.tensor([0.5, 2.0, 5.0], dtype=dtype)
    x = torch.diag_embed(diagonal).expand(batch + (3, 3))
    u = torch.tensor([[1.0, 2.0, -1.0], [2.0, 0.5, 3.0], [-1.0, 3.0, 2.0]], dtype=dtype)
    v = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 3.0, 1.0], [0.5, 1.0, -2.0]], dtype=dtype)
    denominator = diagonal[:, None] * diagonal[None, :]
    for other in [None, v]:
        expected = (u * (u if other is None else other) / denominator).sum()
        result = manifold.inner(x, u, other, keepdim=keepdim)
        assert result.shape == batch + ((1, 1) if keepdim else ())
        torch.testing.assert_close(result, expected.expand_as(result))
    expected_retr = x + u + 0.5 * (u / diagonal) @ u
    torch.testing.assert_close(manifold.retr(x, u), expected_retr)


@pytest.mark.parametrize("same_tangent", [False, True])
def test_metric_analytic_gradients(same_tangent):
    manifold = geoopt.SymmetricPositiveDefinite()
    x = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[1.0, 2.0], [2.0, -3.0]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[2.0, -1.0], [-1.0, 4.0]], dtype=torch.float64, requires_grad=True)
    inverse = torch.tensor([[3.0, -1.0], [-1.0, 4.0]], dtype=torch.float64) / 11
    if same_tangent:
        loss = manifold.inner(x, u)
        gx, gu = torch.autograd.grad(loss, (x, u))
        torch.testing.assert_close(gu, 2 * inverse @ u @ inverse)
        torch.testing.assert_close(gx, -2 * inverse @ u @ inverse @ u @ inverse)
    else:
        loss = manifold.inner(x, u, v)
        gx, gu, gv = torch.autograd.grad(loss, (x, u, v))
        torch.testing.assert_close(gu, inverse @ v @ inverse)
        torch.testing.assert_close(gv, inverse @ u @ inverse)
        expected_x = -(inverse @ u @ inverse @ v @ inverse + inverse @ v @ inverse @ u @ inverse)
        torch.testing.assert_close(gx, expected_x)


def test_retraction_values_and_derivatives_against_inverse_formula():
    manifold = geoopt.SymmetricPositiveDefinite()
    x = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[0.1, 0.2], [0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=torch.float64)
    inverse = torch.linalg.inv(x)
    reference = x + u + 0.5 * u @ inverse @ u
    reference = (reference + reference.T) / 2
    actual = manifold.retr(x, u)
    torch.testing.assert_close(actual, reference)
    expected_grad = torch.autograd.grad((reference * weight).sum(), (x, u))
    actual_grad = torch.autograd.grad((actual * weight).sum(), (x, u))
    for expected, result in zip(expected_grad, actual_grad):
        torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_broadcast_and_noncontiguous_metric_inputs(dtype):
    manifold = geoopt.SymmetricPositiveDefinite()
    generator = torch.Generator().manual_seed(1201)
    x = torch.randn((2, 1, 3, 3), generator=generator, dtype=dtype)
    x = x @ x.transpose(-1, -2) + 2 * torch.eye(3, dtype=dtype)
    u = torch.randn((1, 4, 3, 3), generator=generator, dtype=dtype)
    u = ((u + u.transpose(-1, -2)) / 2).transpose(-1, -2)
    v = torch.randn((2, 4, 3, 3), generator=generator, dtype=dtype)
    v = ((v + v.transpose(-1, -2)) / 2).transpose(-1, -2)
    inverse = torch.linalg.inv(x)
    expected = torch.diagonal(inverse @ u @ inverse @ v, dim1=-2, dim2=-1).sum(-1)
    actual = manifold.inner(x, u, v)
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(manifold.retr(x, u), x + u + 0.5 * u @ inverse @ u)
