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


def test_sabs_is_exact_above_the_floor():
    """``sabs`` must not perturb values that were never near zero."""
    import torch
    from geoopt.utils import sabs

    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, 1e-3, 3.7, 1e-6], dtype=dtype)
        assert torch.equal(sabs(x), x.abs())
        assert torch.equal(sabs(-x), x.abs())


def test_sabs_still_guards_zero():
    import torch
    from geoopt.utils import sabs

    z = torch.zeros((), dtype=torch.float64, requires_grad=True)
    out = sabs(z)
    assert float(out) > 0.0
    out.sqrt().reciprocal().backward()
    assert torch.isfinite(z.grad).all()


def test_project_commutes_with_curvature_rescaling():
    """The ball is the same ball in different units.

    Under ``x -> x / s`` and ``k -> k * s**2`` the Poincare ball of curvature
    ``k`` maps onto the ball of curvature ``k * s**2``, so projecting and then
    rescaling must agree with rescaling and then projecting. With an additive
    floor inside ``sqrt(|k|)`` the two disagree, and the disagreement grows as
    ``|k|`` shrinks towards the floor: about 5e-4 at ``k = -1e-12`` in float64.
    """
    import torch
    from geoopt.manifolds.stereographic import math as smath

    torch.manual_seed(0)
    x0 = torch.randn(32, 8, dtype=torch.float64)
    for k0 in (-1.0, -1e-6, -1e-12):
        k = torch.tensor(k0, dtype=torch.float64)
        # start outside the ball so that the projection actually clips
        x = x0 / x0.norm(dim=-1, keepdim=True) * (1.5 / abs(k0) ** 0.5)
        ref = smath.project(x, k=k)
        for s in (2.0, 4.0, 8.0):
            ks = torch.tensor(k0 * s * s, dtype=torch.float64)
            got = smath.project(x / s, k=ks) * s
            assert torch.allclose(got, ref, rtol=0, atol=0), (
                f"projection is not scale-equivariant at k={k0}, s={s}: "
                f"max |diff| = {(got - ref).abs().max().item():.3e}"
            )
