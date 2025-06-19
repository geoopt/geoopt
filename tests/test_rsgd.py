import geoopt
from geoopt.manifolds.siegel.vvd_metrics import SiegelMetricType
import torch
import numpy as np
from itertools import product
import pytest


@pytest.mark.parametrize(
    "params",
    [
        dict(lr=1e-2),
        dict(lr=1e-3, momentum=0.9),
        dict(momentum=0.9, nesterov=True, lr=1e-3),
        dict(momentum=0.9, dampening=0.1, lr=1e-3),
    ],
)
def test_rsgd_stiefel(params):
    stiefel = geoopt.manifolds.Stiefel()
    torch.manual_seed(42)
    with torch.no_grad():
        X = geoopt.ManifoldParameter(torch.randn(20, 10), manifold=stiefel).proj_()
    Xstar = torch.randn(20, 10)
    Xstar.set_(stiefel.projx(Xstar))

    def closure():
        optim.zero_grad()
        loss = (X - Xstar).pow(2).sum()
        # manifold constraint that makes optimization hard if violated
        loss += (X.t() @ X - torch.eye(X.shape[1])).pow(2).sum() * 100
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianSGD([X], **params)
    assert (X - Xstar).norm() > 1e-5
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-5:
            break
        optim.step(closure)
    assert X.is_contiguous()
    np.testing.assert_allclose(X.data, Xstar, atol=1e-5)
    optim.load_state_dict(optim.state_dict())
    optim.step(closure)


@pytest.mark.parametrize(
    "params",
    [
        dict(lr=1e-2),
        dict(lr=1e-3, momentum=0.9),
        dict(momentum=0.9, nesterov=True, lr=1e-3),
        dict(momentum=0.9, dampening=0.1, lr=1e-3),
    ],
)
def test_rsgd_spd(params):
    manifold = geoopt.manifolds.SymmetricPositiveDefinite()
    torch.manual_seed(42)
    with torch.no_grad():
        X = geoopt.ManifoldParameter(manifold.random(2, 2), manifold=manifold).proj_()
    Xstar = manifold.random(2, 2)
    # Xstar.set_(manifold.projx(Xstar))

    def closure():
        optim.zero_grad()
        loss = (X - Xstar).pow(2).sum()
        # manifold constraint that makes optimization hard if violated
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianSGD([X], **params)
    assert (X - Xstar).norm() > 1e-5
    for i in range(10000):
        cond = (X - Xstar).norm()
        if cond < 1e-5:
            break
        optim.step(closure)
        print(i, cond)
    assert X.is_contiguous()
    np.testing.assert_allclose(X.data, Xstar, atol=1e-5)
    optim.load_state_dict(optim.state_dict())
    optim.step(closure)


@pytest.fixture(
    scope="module",
    params=product(
        [geoopt.manifolds.UpperHalf, geoopt.manifolds.BoundedDomain],
        [
            SiegelMetricType.RIEMANNIAN,
            SiegelMetricType.FINSLER_ONE,
            SiegelMetricType.FINSLER_INFINITY,
            SiegelMetricType.FINSLER_MINIMUM,
            SiegelMetricType.WEIGHTED_SUM,
        ],
    ),
)
def complex_manifold(request):
    manifold_class, metric = request.param
    return manifold_class(metric=metric, rank=2)


@pytest.mark.parametrize(
    "params",
    [
        dict(lr=1e-3),
        dict(lr=1e-3, momentum=0.9),
        dict(momentum=0.9, nesterov=True, lr=1e-3),
        dict(momentum=0.9, dampening=0.1, lr=1e-3),
    ],
)
def test_rsgd_complex_manifold(params, complex_manifold):
    manifold = complex_manifold
    torch.manual_seed(42)
    with torch.no_grad():
        X = geoopt.ManifoldParameter(manifold.random(2, 2), manifold=manifold).proj_()
    Xstar = manifold.random(2, 2)

    def closure():
        optim.zero_grad()
        loss = manifold.dist(X, Xstar).pow(2).sum()
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianSGD([X], **params)
    assert manifold.dist(X, Xstar).item() > 1e-1
    for i in range(10000):
        distance = manifold.dist(X, Xstar).item()
        if distance < 1e-4:
            break
        try:
            optim.step(closure)
        except UserWarning:
            # On the first pass it raises a UserWarning due to discarding part of the
            # complex variable in a casting
            pass
        print(i, distance)
    distance = manifold.dist(X, Xstar).item()
    np.testing.assert_equal(distance < 1e-4, True)
    optim.load_state_dict(optim.state_dict())
    optim.step(closure)


def test_init_manifold():
    torch.manual_seed(42)
    stiefel = geoopt.manifolds.Stiefel()
    rn = geoopt.manifolds.Euclidean()
    x0 = torch.randn(10, 10)
    x1 = torch.randn(10, 10)
    with torch.no_grad():
        p0 = geoopt.ManifoldParameter(x0, manifold=stiefel).proj_()
    p1 = geoopt.ManifoldParameter(x1, manifold=rn)
    p0.grad = torch.zeros_like(p0)
    p1.grad = torch.zeros_like(p1)
    p0old = p0.clone()
    p1old = p1.clone()
    opt = geoopt.optim.RiemannianSGD([p0, p1], lr=1, stabilize=1)
    opt.zero_grad()
    opt.step()
    assert not np.allclose(p0.data, p0old.data)
    assert p0.is_contiguous()
    np.testing.assert_allclose(p1.data, p1old.data)
    np.testing.assert_allclose(p0.data, stiefel.projx(p0old.data), atol=1e-4)
