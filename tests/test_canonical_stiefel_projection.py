import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 2), (4, 2), (4, 1)])
def test_canonical_projection_preserves_tangent_vectors(dtype, shape):
    n, k = shape
    generator = torch.Generator().manual_seed(29)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=dtype, generator=generator))
    x = q[:, :k]
    omega = torch.randn(k, k, dtype=dtype, generator=generator)
    omega = omega - omega.T
    horizontal = torch.randn(n - k, k, dtype=dtype, generator=generator)
    tangent = x @ omega + q[:, k:] @ horizontal
    manifold = geoopt.CanonicalStiefel()

    manifold.assert_check_vector_on_tangent(x, tangent)
    torch.testing.assert_close(manifold.proju(x, tangent), tangent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_canonical_projection_is_idempotent_and_metric_orthogonal(dtype):
    manifold = geoopt.CanonicalStiefel()
    x = torch.eye(4, 2, dtype=dtype)
    ambient = torch.tensor(
        [[[1, 2], [3, 4], [5, 6], [7, 8]], [[3, -2], [1, 5], [0, 2], [1, -1]]],
        dtype=dtype,
    )
    tangent = torch.tensor([[0, -2], [2, 0], [1, 3], [-4, 2]], dtype=dtype)

    projected = manifold.proju(x, ambient)

    assert projected.shape == ambient.shape
    manifold.assert_check_vector_on_tangent(x, projected)
    torch.testing.assert_close(manifold.proju(x, projected), projected)
    torch.testing.assert_close(
        manifold.inner(x, ambient - projected, tangent), torch.zeros(2, dtype=dtype)
    )
    # The canonical gradient has a different vertical scale from a projection.
    gradient = manifold.egrad2rgrad(x, ambient)
    torch.testing.assert_close(
        manifold.inner(x, gradient, tangent), (ambient * tangent).sum((-2, -1))
    )


@pytest.mark.parametrize("optimizer_name", ["RiemannianAdam", "RiemannianSGD"])
def test_stabilizing_canonical_momentum_does_not_rescale_it(optimizer_name):
    manifold = geoopt.CanonicalStiefel()
    point = geoopt.ManifoldParameter(torch.eye(3, 2, dtype=torch.float64), manifold=manifold)
    point.grad = torch.tensor([[0, -1], [1, 0], [2, -3]], dtype=point.dtype)
    optimizer_type = getattr(geoopt.optim, optimizer_name)
    kwargs = {"momentum": 0.5} if optimizer_name == "RiemannianSGD" else {}
    optimizer = optimizer_type([point], lr=0, **kwargs)
    optimizer.step()
    state_key = "momentum_buffer" if optimizer_name == "RiemannianSGD" else "exp_avg"
    expected = optimizer.state[point][state_key].clone()
    manifold.assert_check_vector_on_tangent(point, expected)

    optimizer.stabilize()
    optimizer.stabilize()

    torch.testing.assert_close(optimizer.state[point][state_key], expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 2), (4, 2), (4, 1), (2, 4, 2)])
def test_canonical_retraction_has_identity_differential(dtype, shape):
    manifold = geoopt.CanonicalStiefel()
    n, k = shape[-2:]
    batch_shape = shape[:-2]
    generator = torch.Generator().manual_seed(29)
    q, _ = torch.linalg.qr(
        torch.randn(*batch_shape, n, n, dtype=dtype, generator=generator)
    )
    x = q[..., :k]
    omega = torch.randn(*batch_shape, k, k, dtype=dtype, generator=generator)
    omega = omega - omega.transpose(-1, -2)
    horizontal = torch.randn(
        *batch_shape, n - k, k, dtype=dtype, generator=generator
    )
    tangent = x @ omega + q[..., k:] @ horizontal
    manifold.assert_check_vector_on_tangent(x, tangent)

    point, velocity = torch.autograd.functional.jvp(
        lambda step: manifold.retr(x, step), torch.zeros_like(tangent), tangent
    )

    torch.testing.assert_close(point, x)
    torch.testing.assert_close(velocity, tangent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (2, 4, 2)])
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_canonical_retraction_jvp_matches_autograd(dtype, shape, scale):
    manifold = geoopt.CanonicalStiefel()
    if scale != 1:
        manifold = geoopt.Scaled(manifold, scale).to(dtype=dtype)
    n, k = shape[-2:]
    batch_shape = shape[:-2]
    generator = torch.Generator().manual_seed(31)
    q, _ = torch.linalg.qr(
        torch.randn(*batch_shape, n, n, dtype=dtype, generator=generator)
    )
    x = q[..., :k]

    def tangent():
        omega = torch.randn(*batch_shape, k, k, dtype=dtype, generator=generator)
        omega = omega - omega.transpose(-1, -2)
        horizontal = torch.randn(
            *batch_shape, n - k, k, dtype=dtype, generator=generator
        )
        return 0.2 * (x @ omega + q[..., k:] @ horizontal)

    u, v = tangent(), tangent()
    point, expected = torch.autograd.functional.jvp(
        lambda step: manifold.retr(x, step), u, v
    )

    actual = manifold.retr_jvp(x, u, v)

    torch.testing.assert_close(actual, expected)
    manifold.assert_check_vector_on_tangent(point, actual)
