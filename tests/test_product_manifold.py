from geoopt import ProductManifold, Sphere, Euclidean, PoincareBall
import torch
import numpy as np
import pytest


def test_init():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)))
    assert pman.n_elements == (10 + 3 * 2)
    assert pman.name == "(Sphere)x(Sphere)"


def test_from_point():
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    pman = ProductManifold.from_point(*point, batch_dims=1)
    assert pman.n_elements == (10 + 3 * 2 + 1)


def test_from_point_checks_shapes():
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(3, 3, 2),
        Euclidean().random_normal(5),
    ]
    pman = ProductManifold.from_point(*point)
    assert pman.n_elements == (5 * 10 + 3 * 3 * 2 + 5 * 1)
    with pytest.raises(ValueError) as e:
        _ = ProductManifold.from_point(*point, batch_dims=1)
    assert e.match("Not all parts have same batch shape")


def test_reshaping():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)), (Euclidean(), ()))
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    tensor = pman.pack_point(*point)
    assert tensor.shape == (5, 10 + 3 * 2 + 1)
    point_new = pman.unpack_tensor(tensor)
    for old, new in zip(point, point_new):
        np.testing.assert_allclose(old, new)


def test_inner_product():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)), (Euclidean(), ()))
    point = [
        Sphere().random_uniform(5, 10),
        Sphere().random_uniform(5, 3, 2),
        Euclidean().random_normal(5),
    ]
    tensor = pman.pack_point(*point)
    tangent = torch.randn_like(tensor)
    tangent = pman.proju(tensor, tangent)

    inner = pman.inner(tensor, tangent)
    assert inner.shape == (5,)
    inner_kd = pman.inner(tensor, tangent, keepdim=True)
    assert inner_kd.shape == (5, 1)


def test_dtype_checked_properly():
    p1 = PoincareBall()
    p2 = PoincareBall().double()
    with pytest.raises(ValueError) as e:
        _ = ProductManifold((p1, (10,)), (p2, (12,)))
    assert e.match("Not all manifold share the same dtype")
