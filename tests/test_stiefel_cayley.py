import pytest
import torch

import geoopt


def dense_cayley(x, u, v):
    adjusted = u - 0.5 * x @ (x.transpose(-1, -2) @ u)
    a = adjusted @ x.transpose(-1, -2) - x @ adjusted.transpose(-1, -2)
    eye = torch.eye(x.shape[-2], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(eye - 0.5 * a, (eye + 0.5 * a) @ v)


def inputs(n, m, batch=(), dtype=torch.float64):
    generator = torch.Generator().manual_seed(8451)
    x = torch.linalg.qr(torch.randn(*batch, n, m, dtype=dtype, generator=generator))[0]
    manifold = geoopt.CanonicalStiefel()
    u = manifold.proju(x, torch.randn(x.shape, dtype=dtype, generator=generator) * 0.1)
    v = manifold.proju(x, torch.randn(x.shape, dtype=dtype, generator=generator))
    return x, u, v


@pytest.mark.parametrize("n,m", [(3, 1), (9, 2), (12, 3), (6, 3), (5, 3), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cayley_values_and_geometry(n, m, dtype):
    manifold = geoopt.CanonicalStiefel()
    x, u, v = inputs(n, m, (2,), dtype)
    new_x, new_v = manifold.retr_transp(x, u, v)
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(new_x, dense_cayley(x, u, x), atol=tol, rtol=tol)
    torch.testing.assert_close(new_v, dense_cayley(x, u, v), atol=tol, rtol=tol)
    torch.testing.assert_close(new_x, manifold.retr(x, u), atol=tol, rtol=tol)
    torch.testing.assert_close(new_v, manifold.transp_follow_retr(x, u, v), atol=tol, rtol=tol)
    assert manifold.check_point_on_manifold(new_x, atol=tol, rtol=tol)
    assert manifold.check_vector_on_tangent(new_x, new_v, atol=tol, rtol=tol)
    torch.testing.assert_close(
        manifold.inner(new_x, new_v), manifold.inner(x, v), atol=tol, rtol=tol
    )


@pytest.mark.parametrize("n,m", [(9, 2), (5, 3)])
@pytest.mark.parametrize("zero_step", [False, True])
def test_cayley_derivatives_match_dense_equations(n, m, zero_step):
    x, u, v = inputs(n, m, (2,))
    if zero_step:
        u.zero_()
    x.requires_grad_()
    u.requires_grad_()
    v.requires_grad_()
    weights = torch.arange(v.numel(), dtype=v.dtype).reshape(v.shape) / v.numel()
    actual = geoopt.CanonicalStiefel().transp_follow_retr(x, u, v)
    expected = dense_cayley(x, u, v)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (x, u, v))
    expected_grad = torch.autograd.grad((expected * weights).sum(), (x, u, v))
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        assert torch.isfinite(actual_value).all()
        torch.testing.assert_close(actual_value, expected_value, atol=1e-10, rtol=1e-10)


def test_cayley_transport_broadcasts_independent_batch_axes():
    manifold = geoopt.CanonicalStiefel()
    x, _, _ = inputs(9, 2, (2, 1))
    _, u, _ = inputs(9, 2, (1, 3))
    _, _, v = inputs(9, 2, (2, 3))
    actual = manifold.transp_follow_retr(x, u, v)
    expected = dense_cayley(x, u, v)
    assert actual.shape == (2, 3, 9, 2)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(
                actual[i, j], dense_cayley(x[i, 0], u[0, j], v[i, j]), atol=1e-10, rtol=1e-10
            )


def test_cayley_handles_noncontiguous_and_empty_batches():
    manifold = geoopt.CanonicalStiefel()
    x, u, v = inputs(9, 2, (4,))
    x, u, v = x[::2], u[::2], v[::2]
    assert not x.is_contiguous()
    torch.testing.assert_close(
        manifold.transp_follow_retr(x, u, v), dense_cayley(x, u, v), atol=1e-10, rtol=1e-10
    )
    assert manifold.retr(x[:0], u[:0]).shape == (0, 9, 2)


def test_cayley_retraction_has_the_requested_initial_velocity():
    manifold = geoopt.CanonicalStiefel()
    x, u, _ = inputs(12, 3, (2,))
    zero = torch.zeros_like(u)
    actual, derivative = torch.autograd.functional.jvp(
        lambda step: manifold.retr(x, step), (zero,), (u,)
    )
    torch.testing.assert_close(actual, x, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(derivative, u, atol=1e-10, rtol=1e-10)
