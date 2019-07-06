import pytest
import torch
import geoopt


def test_random_R():
    manifold = geoopt.Euclidean()
    point = manifold.random_normal(10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Euclidean():
    manifold = geoopt.Euclidean(ndim=1)
    point = manifold.random_normal(10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Stiefel():
    manifold = geoopt.Stiefel()
    point = manifold.random_naive(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Sphere():
    manifold = geoopt.Sphere()
    point = manifold.random_uniform(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_SphereProjection():
    subspace = torch.rand(10, 2, dtype=torch.float64)
    manifold = geoopt.Sphere(intersection=subspace)
    point = manifold.random_uniform(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_random_Poincare():
    manifold = geoopt.PoincareBall()
    point = manifold.random_normal(3, 10, 10)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_fails_Euclidean():
    with pytest.raises(ValueError):
        manifold = geoopt.Euclidean(ndim=1)
        manifold.random_normal(())


def test_fails_Stiefel():
    with pytest.raises(ValueError):
        manifold = geoopt.Stiefel()
        manifold.random_naive(())
    with pytest.raises(ValueError):
        manifold = geoopt.Stiefel()
        manifold.random_naive((5, 10))


def test_fails_Sphere():
    with pytest.raises(ValueError):
        manifold = geoopt.Sphere()
        manifold.random_uniform(())
    with pytest.raises(ValueError):
        manifold = geoopt.Sphere()
        manifold.random_uniform(1)


def test_fails_SphereProjection():
    subspace = torch.rand(10, 2, dtype=torch.float64)
    manifold = geoopt.Sphere(intersection=subspace)
    with pytest.raises(ValueError):
        manifold.random_uniform(50)


def test_fails_Poincare():
    with pytest.raises(ValueError):
        manifold = geoopt.PoincareBall()
        manifold.random_normal(())
