import pytest
import torch
import geoopt


def test_random_R():
    manifold = geoopt.Euclidean()
    point = manifold.origin(10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Euclidean():
    manifold = geoopt.Euclidean(ndim=1)
    point = manifold.origin(10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Stiefel():
    manifold = geoopt.Stiefel()
    point = manifold.origin(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_BirkhoffPolytope():
    manifold = geoopt.BirkhoffPolytope()
    point = manifold.origin(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Sphere():
    manifold = geoopt.Sphere()
    point = manifold.origin(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_SphereProjection():
    subspace = torch.rand(10, 2, dtype=torch.float64)
    manifold = geoopt.Sphere(intersection=subspace)
    point = manifold.origin(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Poincare():
    manifold = geoopt.PoincareBall()
    point = manifold.origin(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Upper_half():
    manifold = geoopt.UpperHalf()
    point = manifold.origin(10, 10)
    manifold.assert_check_point_on_manifold(point)


def test_fails_Euclidean():
    with pytest.raises(ValueError):
        manifold = geoopt.Euclidean(ndim=1)
        manifold.origin(())


def test_fails_Stiefel():
    with pytest.raises(ValueError):
        manifold = geoopt.Stiefel()
        manifold.origin(())
    with pytest.raises(ValueError):
        manifold = geoopt.Stiefel()
        manifold.origin((5, 10))


def test_fails_Sphere():
    with pytest.raises(ValueError):
        manifold = geoopt.Sphere()
        manifold.origin(())
    with pytest.raises(ValueError):
        manifold = geoopt.Sphere()
        manifold.origin(1)


def test_fails_SphereProjection():
    subspace = torch.rand(10, 2, dtype=torch.float64)
    manifold = geoopt.Sphere(intersection=subspace)
    with pytest.raises(ValueError):
        manifold.origin(50)


def test_fails_Poincare():
    with pytest.raises(ValueError):
        manifold = geoopt.PoincareBall()
        manifold.origin(())


def test_product():
    manifold = geoopt.ProductManifold(
        (geoopt.Sphere(), 10),
        (geoopt.PoincareBall(), 3),
        (geoopt.Stiefel(), (20, 2)),
        (geoopt.Euclidean(), 43),
    )
    sample = manifold.origin(20, manifold.n_elements)
    manifold.assert_check_point_on_manifold(sample)
