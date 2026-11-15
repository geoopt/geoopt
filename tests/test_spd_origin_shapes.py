import pytest
import torch

import geoopt


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (2, 4, 3, 3)])
@pytest.mark.parametrize("shape_type", [tuple, torch.Size])
def test_spd_origin_accepts_packed_shape(shape, shape_type):
    manifold = geoopt.SymmetricPositiveDefinite()

    actual = manifold.origin(shape_type(shape), dtype=torch.float64, device="cpu")

    expected = torch.eye(3, dtype=torch.float64).expand(shape)
    torch.testing.assert_close(actual, expected)
    manifold.assert_check_point_on_manifold(actual)


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (2, 4, 3, 3)])
def test_spd_origin_variadic_shape_control(shape):
    manifold = geoopt.SymmetricPositiveDefinite()
    actual = manifold.origin(*shape, dtype=torch.float64)
    torch.testing.assert_close(actual, torch.eye(3, dtype=torch.float64).expand(shape))


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_product_origin_initializes_spd_and_euclidean_factors(batch_shape):
    manifold = geoopt.ProductManifold(
        (geoopt.SymmetricPositiveDefinite(), (2, 2)),
        (geoopt.Euclidean(ndim=1), 3),
    )

    actual = manifold.origin(*batch_shape, 7, dtype=torch.float64)

    expected = torch.cat((torch.eye(2, dtype=torch.float64).reshape(-1), torch.zeros(3)))
    torch.testing.assert_close(actual, expected.expand(*batch_shape, 7))
    assert actual.manifold is manifold
    manifold.assert_check_point_on_manifold(actual)


def test_nested_product_origin_initializes_each_spd_factor():
    inner = geoopt.ProductManifold(
        (geoopt.SymmetricPositiveDefinite(), (2, 2)),
        (geoopt.Euclidean(ndim=1), 2),
    )
    manifold = geoopt.ProductManifold((inner, 6), (geoopt.SymmetricPositiveDefinite(), (3, 3)))

    actual = manifold.origin(4, 15, dtype=torch.float64)

    expected = torch.cat(
        (torch.eye(2, dtype=torch.float64).reshape(-1), torch.zeros(2), torch.eye(3).reshape(-1))
    )
    torch.testing.assert_close(actual, expected.expand(4, 15))
    manifold.assert_check_point_on_manifold(actual)


def test_product_origin_can_initialize_a_trainable_spd_parameter():
    manifold = geoopt.ProductManifold(
        (geoopt.SymmetricPositiveDefinite(), (2, 2)),
        (geoopt.Euclidean(ndim=1), 3),
    )
    point = geoopt.ManifoldParameter(manifold.origin(2, 7, dtype=torch.float64))
    target = manifold.pack_point(
        1.25 * torch.eye(2, dtype=torch.float64).expand(2, 2, 2),
        torch.full((2, 3), 0.1, dtype=torch.float64),
    )
    optimizer = geoopt.optim.RiemannianSGD([point], lr=0.05)
    initial_loss = (point - target).square().sum().item()

    for _ in range(3):
        optimizer.zero_grad()
        loss = (point - target).square().sum()
        loss.backward()
        optimizer.step()

    assert (point - target).square().sum().item() < initial_loss
    manifold.assert_check_point_on_manifold(point)
