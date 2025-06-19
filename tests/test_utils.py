import pytest
import torch.nn
import numpy as np
import geoopt
import tempfile
import os


@pytest.fixture
def A():
    torch.manual_seed(42)
    n = 10
    a = torch.randn(n, 3, 3).double()
    a[:, 2, :] = 0
    return a.clone().requires_grad_()


def test_expm(A):
    from scipy.linalg import expm
    import numpy as np

    expm_scipy = np.zeros_like(A.detach())
    for i in range(A.shape[0]):
        expm_scipy[i] = expm(A.detach()[i].numpy())
    expm_torch = geoopt.linalg.expm(A)
    np.testing.assert_allclose(expm_torch.detach(), expm_scipy, rtol=1e-6, atol=1e-13)
    expm_torch.sum().backward()  # this should work


def test_pickle1():
    t = torch.ones(10)
    p = geoopt.ManifoldTensor(t, manifold=geoopt.Sphere())
    with tempfile.TemporaryDirectory() as path:
        torch.save(p, os.path.join(path, "tens.t7"))
        p1 = torch.load(os.path.join(path, "tens.t7"), weights_only=False)
    assert isinstance(p1, geoopt.ManifoldTensor)
    assert p.stride() == p1.stride()
    assert p.storage_offset() == p1.storage_offset()
    assert p.requires_grad == p1.requires_grad
    np.testing.assert_allclose(p.detach(), p1.detach())
    assert isinstance(p.manifold, type(p1.manifold))


def test_pickle2():
    t = torch.ones(10)
    p = geoopt.ManifoldParameter(t, manifold=geoopt.Sphere())
    with tempfile.TemporaryDirectory() as path:
        torch.save(p, os.path.join(path, "tens.t7"))
        p1 = torch.load(os.path.join(path, "tens.t7"))
    assert isinstance(p1, geoopt.ManifoldParameter)
    assert p.stride() == p1.stride()
    assert p.storage_offset() == p1.storage_offset()
    assert p.requires_grad == p1.requires_grad
    np.testing.assert_allclose(p.detach(), p1.detach())
    assert isinstance(p.manifold, type(p1.manifold))


def test_pickle3():
    t = torch.ones(10)
    span = torch.randn(10, 2)
    sub_sphere = geoopt.manifolds.Sphere(intersection=span)
    p = geoopt.ManifoldParameter(t, manifold=sub_sphere)
    with tempfile.TemporaryDirectory() as path:
        torch.save(p, os.path.join(path, "tens.t7"))
        p1 = torch.load(os.path.join(path, "tens.t7"))
    assert isinstance(p1, geoopt.ManifoldParameter)
    assert p.stride() == p1.stride()
    assert p.storage_offset() == p1.storage_offset()
    assert p.requires_grad == p1.requires_grad
    np.testing.assert_allclose(p.detach(), p1.detach())
    assert isinstance(p.manifold, type(p1.manifold))
    np.testing.assert_allclose(p.manifold.projector, p1.manifold.projector)


def test_manifold_to_smth():
    span = torch.randn(10, 2)
    sub_sphere = geoopt.manifolds.Sphere(intersection=span)
    sub_sphere.to(torch.float64)
    assert sub_sphere.projector.dtype == torch.float64


def test_manifold_is_submodule():
    span = torch.randn(10, 2)
    sub_sphere = geoopt.manifolds.Sphere(intersection=span)
    sub_sphere.to(torch.float64)
    container = torch.nn.ModuleDict({"sphere": sub_sphere})
    container.to(torch.float64)
    assert sub_sphere.projector.dtype == torch.float64


def test_manifold_is_submodule_poincare():
    c = torch.tensor(1.0)
    ball = geoopt.manifolds.PoincareBall(c)
    assert ball.c.dtype == torch.float32
    ball.to(torch.float64)
    container = torch.nn.ModuleDict({"ball": ball})
    container.to(torch.float64)
    assert ball.c.dtype == torch.float64


def test_ismanifold():
    m1 = geoopt.Euclidean()
    assert geoopt.ismanifold(m1, geoopt.Euclidean)
    m1 = geoopt.Scaled(m1)
    m1 = geoopt.Scaled(m1)
    assert geoopt.ismanifold(m1, geoopt.Euclidean)

    with pytest.raises(TypeError):
        geoopt.ismanifold(m1, int)

    with pytest.raises(TypeError):
        geoopt.ismanifold(m1, 1)

    assert not geoopt.ismanifold(1, geoopt.Euclidean)
