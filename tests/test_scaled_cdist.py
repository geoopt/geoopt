import pytest
import torch

import geoopt


def pairwise_example(kind):
    generator = torch.Generator().manual_seed(1729)
    x = torch.randn(2, 3, 4, dtype=torch.float64, generator=generator)
    y = torch.randn(1, 4, 4, dtype=torch.float64, generator=generator) + 2
    if kind == "scalar":
        manifold, x, y = geoopt.Euclidean(ndim=0), x[..., 0], y[..., 0]
    elif kind == "vector":
        manifold = geoopt.Euclidean(ndim=1)
    elif kind == "matrix":
        manifold = geoopt.Euclidean(ndim=2)
        x, y = x.reshape(2, 3, 2, 2), y.reshape(1, 4, 2, 2)
    elif kind == "sphere":
        manifold = geoopt.Sphere()
        x, y = manifold.projx(x), manifold.projx(y)
    else:
        manifold = geoopt.Lorentz(k=torch.tensor(1.5, dtype=torch.float64))
        x = torch.cat(((manifold.k + x.square().sum(-1, keepdim=True)).sqrt(), x), -1)
        y = torch.cat(((manifold.k + y.square().sum(-1, keepdim=True)).sqrt(), y), -1)
    x = x.transpose(0, 1).contiguous().transpose(0, 1)
    return manifold, x, y


@pytest.mark.parametrize("kind", ["scalar", "vector", "matrix", "sphere", "lorentz"])
@pytest.mark.parametrize("scale", [0.5, 3.0])
def test_scaled_cdist_matches_individual_distances(kind, scale):
    base, x, y = pairwise_example(kind)
    manifold = geoopt.Scaled(base, scale).double()
    expected = torch.stack(
        [
            torch.stack([manifold.dist(x[:, i], y[:, j]) for j in range(4)], dim=-1)
            for i in range(3)
        ],
        dim=-2,
    )

    actual = manifold.cdist(x, y)

    assert not x.is_contiguous()
    assert actual.shape == (2, 3, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_pairwise_loss_differentiates_embeddings_and_learnable_scale():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[-2.0, 1.0], [-1.0, -3.0]], dtype=torch.float64, requires_grad=True)
    manifold = geoopt.Scaled(geoopt.Euclidean(ndim=1), 2.0, learnable=True).double()

    loss = manifold.cdist(x, y).square().sum()
    grad_x, grad_y, grad_scale = torch.autograd.grad(loss, (x, y, manifold._log_scale))

    scale_sq = manifold.scale.detach().square()
    expected_loss = scale_sq * (x[:, None] - y[None, :]).square().sum()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(grad_x, 2 * scale_sq * (len(y) * x - y.sum(0)))
    torch.testing.assert_close(grad_y, 2 * scale_sq * (len(x) * y - x.sum(0)))
    torch.testing.assert_close(grad_scale, 2 * expected_loss)


def test_nested_scales_both_receive_pairwise_loss_gradients():
    base, x, y = pairwise_example("vector")
    inner = geoopt.Scaled(base, 2.0, learnable=True).double()
    outer = geoopt.Scaled(inner, 3.0, learnable=True).double()

    actual = outer.cdist(x, y)
    expected = inner.scale * outer.scale * base.cdist(x, y)
    torch.testing.assert_close(actual, expected)
    loss = actual.sum()
    gradients = torch.autograd.grad(loss, (inner._log_scale, outer._log_scale))
    for gradient in gradients:
        torch.testing.assert_close(gradient, loss)


def test_scaled_cdist_preserves_empty_batches():
    base, x, y = pairwise_example("sphere")
    manifold = geoopt.Scaled(base, 2.0).double()
    assert manifold.cdist(x[:, :0], y).shape == (2, 0, 4)


def test_unsupported_base_still_refuses_pairwise_distance():
    manifold = geoopt.Scaled(geoopt.Stiefel(), 2.0)
    with pytest.raises(NotImplementedError):
        manifold.cdist(torch.eye(2).unsqueeze(0), torch.eye(2).unsqueeze(0))


def test_unscaled_pairwise_distance_control():
    manifold = geoopt.Euclidean(ndim=1)
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
    y = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(manifold.cdist(x, y), torch.tensor([[3.0], [4.0]], dtype=torch.float64))
