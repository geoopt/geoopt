from geoopt import ProductManifold, Sphere, Euclidean
import torch
import numpy as np


def test_init():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)))
    assert pman.n_elements == (10 + 3 * 2)
    assert pman.name == "(Sphere)x(Sphere)"


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
