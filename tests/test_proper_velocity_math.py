import numpy as np
import pytest
import torch

import geoopt
from geoopt.manifolds.lorentz import math as lmath
from geoopt.manifolds.proper_velocity import math as pvmath


@pytest.fixture(autouse=True, params=[1, 2, 3])
def seed(request):
    torch.manual_seed(request.param)
    yield


@pytest.fixture(autouse=True)
def use_float64():
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype_old)


@pytest.fixture(params=[0.5, 1.0, 3.0])
def k(request):
    return torch.tensor(request.param)


def lift_vector(x, v, k):
    """Push a PV tangent vector forward to the hyperboloid."""
    x0 = torch.sqrt(k + (x * x).sum(-1, keepdim=True))
    return torch.cat(((x * v).sum(-1, keepdim=True) / x0, v), dim=-1)


def test_lorentz_isometry(k):
    x, y = torch.randn(16, 4) * 2, torch.randn(16, 4) * 2
    u, v = torch.randn(16, 4), torch.randn(16, 4)
    X, Y = pvmath.pv_to_lorentz(x, k=k), pvmath.pv_to_lorentz(y, k=k)
    lorentz = geoopt.Lorentz(k=k)
    lorentz.assert_check_point_on_manifold(X)

    np.testing.assert_allclose(pvmath.dist(x, y, k=k), lmath.dist(X, Y, k=k), rtol=1e-10)
    np.testing.assert_allclose(pvmath.dist0(y, k=k), lmath.dist0(Y, k=k), rtol=1e-10)
    np.testing.assert_allclose(
        pvmath.inner(x, u, v, k=k),
        lmath.inner(lift_vector(x, u, k), lift_vector(x, v, k)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pvmath.pv_to_lorentz(pvmath.expmap(x, u, k=k), k=k),
        lmath.expmap(X, lift_vector(x, u, k), k=k),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        lift_vector(x, pvmath.logmap(x, y, k=k), k), lmath.logmap(X, Y, k=k), atol=1e-8
    )
    np.testing.assert_allclose(
        lift_vector(y, pvmath.parallel_transport(x, y, u, k=k), k),
        lmath.parallel_transport(X, Y, lift_vector(x, u, k), k=k),
        atol=1e-8,
    )
    np.testing.assert_allclose(pvmath.lorentz_to_pv(X), x)


def test_poincare_isometry(k):
    x, y = torch.randn(16, 4) * 2, torch.randn(16, 4) * 2
    ball = geoopt.PoincareBall(c=1 / k)
    p, q = pvmath.pv_to_poincare(x, k=k), pvmath.pv_to_poincare(y, k=k)
    ball.assert_check_point_on_manifold(p)
    np.testing.assert_allclose(pvmath.dist(x, y, k=k), ball.dist(p, q), rtol=1e-8)
    np.testing.assert_allclose(pvmath.poincare_to_pv(p, k=k), x, rtol=1e-8, atol=1e-10)
    # the isomorphism also carries gyroaddition
    np.testing.assert_allclose(
        pvmath.pv_to_poincare(pvmath.gyroadd(x, y, k=k), k=k),
        ball.mobius_add(p, q),
        atol=1e-8,
    )


def test_egrad2rgrad_is_inverse_metric(k):
    x, grad, w = torch.randn(16, 4) * 2, torch.randn(16, 4), torch.randn(16, 4)
    rgrad = pvmath.egrad2rgrad(x, grad, k=k)
    np.testing.assert_allclose(pvmath.inner(x, rgrad, w, k=k), (grad * w).sum(-1), atol=1e-10)


def test_gyro_identities(k):
    x, y = torch.randn(16, 4) * 2, torch.randn(16, 4) * 2
    zero = torch.zeros(16, 4)
    np.testing.assert_allclose(pvmath.gyroadd(zero, x, k=k), x)
    np.testing.assert_allclose(pvmath.gyroadd(x, zero, k=k), x, atol=1e-12)
    np.testing.assert_allclose(pvmath.gyroadd(pvmath.gyroinv(x), x, k=k), zero, atol=1e-10)
    # left cancellation
    np.testing.assert_allclose(
        pvmath.gyroadd(-x, pvmath.gyroadd(x, y, k=k), k=k), y, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(pvmath.gyroscalar(torch.tensor(1.0), x, k=k), x, atol=1e-10)
    np.testing.assert_allclose(pvmath.gyroscalar(torch.tensor(-1.0), x, k=k), -x, atol=1e-10)
    # gyrations are isometries of R^n
    np.testing.assert_allclose(
        pvmath.gyration(x, y, zero + 1.0, k=k).norm(dim=-1),
        (zero + 1.0).norm(dim=-1),
        rtol=1e-8,
    )


def test_gyro_by_riemannian(k):
    # x ⊕ y = Exp_x(PT_{0->x}(Log_0(y))) and t ⊗ x = Exp_0(t Log_0(x))
    x, y = torch.randn(16, 4) * 2, torch.randn(16, 4) * 2
    t = torch.randn(16, 1)
    np.testing.assert_allclose(
        pvmath.gyroadd(x, y, k=k),
        pvmath.expmap(x, pvmath.parallel_transport0(x, pvmath.logmap0(y, k=k), k=k), k=k),
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pvmath.gyroscalar(t, x, k=k),
        pvmath.expmap0(t * pvmath.logmap0(x, k=k), k=k),
        atol=1e-10,
    )


def test_parallel_transport(k):
    x, y = torch.randn(16, 4), torch.randn(16, 4)
    u, w = torch.randn(16, 4), torch.randn(16, 4)
    pu = pvmath.parallel_transport(x, y, u, k=k)
    pw = pvmath.parallel_transport(x, y, w, k=k)
    np.testing.assert_allclose(pvmath.inner(y, pu, pw, k=k), pvmath.inner(x, u, w, k=k), atol=1e-10)
    # composition through the origin and the gyration
    composed = pvmath.parallel_transport0(
        y,
        pvmath.gyration(y, -x, pvmath.parallel_transport0back(x, u, k=k), k=k),
        k=k,
    )
    np.testing.assert_allclose(pu, composed, atol=1e-8)
    np.testing.assert_allclose(pvmath.parallel_transport(y, x, pu, k=k), u, atol=1e-10)
    np.testing.assert_allclose(
        pvmath.parallel_transport0back(y, pvmath.parallel_transport0(y, u, k=k), k=k),
        u,
        atol=1e-10,
    )
    zero = torch.zeros_like(x)
    np.testing.assert_allclose(
        pvmath.parallel_transport(zero, y, u, k=k),
        pvmath.parallel_transport0(y, u, k=k),
        atol=1e-12,
    )


@pytest.mark.parametrize("scale", [1e1, 1e2, 1e3])
def test_parallel_transport_stable_at_large_norm(scale, k):
    x, y = torch.randn(16, 4) * scale, torch.randn(16, 4) * scale
    u, w = torch.randn(16, 4), torch.randn(16, 4)
    pu = pvmath.parallel_transport(x, y, u, k=k)
    pw = pvmath.parallel_transport(x, y, w, k=k)
    np.testing.assert_allclose(
        pvmath.inner(y, pu, pw, k=k), pvmath.inner(x, u, w, k=k), rtol=1e-8, atol=1e-8
    )


def test_geodesic(k):
    x, y = torch.randn(16, 4) * 2, torch.randn(16, 4) * 2
    t = torch.rand(16, 1)
    np.testing.assert_allclose(pvmath.geodesic(torch.zeros(16, 1), x, y, k=k), x, atol=1e-10)
    np.testing.assert_allclose(
        pvmath.geodesic(torch.ones(16, 1), x, y, k=k), y, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        pvmath.dist(x, pvmath.geodesic(t, x, y, k=k), k=k),
        t.squeeze(-1) * pvmath.dist(x, y, k=k),
        rtol=1e-8,
        atol=1e-10,
    )


def test_frechet_mean(k):
    x = torch.randn(32, 4)
    mean = pvmath.frechet_mean(x, k=k, tol=1e-12)
    assert mean.shape == (4,)
    # first-order optimality condition
    np.testing.assert_allclose(pvmath.logmap(mean, x, k=k).mean(0), 0.0, atol=1e-9)
    # homogeneity under left gyrotranslation
    b = torch.randn(4)
    np.testing.assert_allclose(
        pvmath.frechet_mean(pvmath.gyroadd(b, x, k=k), k=k, tol=1e-12),
        pvmath.gyroadd(b, mean, k=k),
        atol=1e-8,
    )
    # two points: the geodesic midpoint
    pair = torch.randn(2, 4)
    np.testing.assert_allclose(
        pvmath.frechet_mean(pair, k=k, tol=1e-12),
        pvmath.geodesic(torch.tensor(0.5), pair[0], pair[1], k=k),
        atol=1e-8,
    )


def test_frechet_mean_reducedim(k):
    x = torch.randn(8, 5, 3)
    mean = pvmath.frechet_mean(x, k=k, reducedim=[0], tol=1e-12)
    assert mean.shape == (5, 3)
    for i in range(5):
        np.testing.assert_allclose(mean[i], pvmath.frechet_mean(x[:, i], k=k, tol=1e-12), atol=1e-8)
    var = pvmath.frechet_variance(x, mean, k=k, reducedim=[0])
    assert var.shape == (5,)
    # the manifold dimension is ignored in reducedim, as in the Lorentz model
    np.testing.assert_allclose(
        pvmath.frechet_mean(x, k=k, reducedim=[0, 2], tol=1e-12), mean, atol=1e-10
    )


def test_frechet_statistics_without_reduction(k):
    x = torch.randn(4)
    np.testing.assert_allclose(pvmath.frechet_mean(x, k=k), x)
    np.testing.assert_allclose(pvmath.frechet_variance(x, x, k=k), 0.0, atol=1e-12)
    y = torch.randn(5, 4)
    np.testing.assert_allclose(pvmath.frechet_mean(y, k=k, reducedim=[]), y)


def test_frechet_variance_homogeneity(k):
    # (1/N) sum d^2(t ⊗ x_i, 0) = t^2 (1/N) sum d^2(x_i, 0)
    x = torch.randn(32, 4)
    t = torch.tensor(2.5)
    zero = torch.zeros(4)
    np.testing.assert_allclose(
        pvmath.frechet_variance(pvmath.gyroscalar(t, x, k=k), zero, k=k),
        t**2 * pvmath.frechet_variance(x, zero, k=k),
        rtol=1e-10,
    )


def test_gradients_at_origin(k):
    x = torch.zeros(3, 4, requires_grad=True)
    out = (
        pvmath.expmap0(x, k=k).sum()
        + pvmath.logmap0(x, k=k).sum()
        + pvmath.gyroscalar(torch.tensor(2.0), x, k=k).sum()
    )
    out.backward()
    assert torch.isfinite(x.grad).all()
    u = torch.zeros(3, 4, requires_grad=True)
    pvmath.norm(torch.randn(3, 4), u, k=k).sum().backward()
    np.testing.assert_allclose(u.grad, 0.0)


def test_learnable_curvature_gradient():
    x, y = torch.randn(8, 4), torch.randn(8, 4)
    manifold = geoopt.ProperVelocity(k=0.7, learnable=True)
    manifold.dist(x, y).sum().backward()
    eps = 1e-6
    numeric = (
        pvmath.dist(x, y, k=torch.tensor(0.7 + eps)).sum()
        - pvmath.dist(x, y, k=torch.tensor(0.7 - eps)).sum()
    ) / (2 * eps)
    np.testing.assert_allclose(manifold.k.grad, numeric, rtol=1e-6)


def test_gyro_operations_match_poincare(k):
    x, y = torch.randn(16, 4) * 2, torch.randn(16, 4) * 2
    v = torch.randn(16, 4)
    t = torch.randn(16, 1)
    ball = geoopt.PoincareBall(c=1 / k)
    p, q = pvmath.pv_to_poincare(x, k=k), pvmath.pv_to_poincare(y, k=k)
    np.testing.assert_allclose(
        pvmath.pv_to_poincare(pvmath.gyroscalar(t, x, k=k), k=k),
        ball.mobius_scalar_mul(t, p),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        pvmath.pv_to_poincare(pvmath.geodesic(t.abs(), x, y, k=k), k=k),
        ball.geodesic(t.abs(), p, q),
        atol=1e-8,
    )
    # gyrations are linear and commute with the radial isometry
    np.testing.assert_allclose(
        pvmath.gyration(x, y, v, k=k),
        geoopt.manifolds.stereographic.math.gyration(p, q, v, k=-1 / k),
        atol=1e-8,
    )


def test_gyration_definition_and_large_norms(k):
    a, b, x = torch.randn(16, 4), torch.randn(16, 4), torch.randn(16, 4)
    definition = pvmath.gyroadd(
        -pvmath.gyroadd(a, b, k=k),
        pvmath.gyroadd(a, pvmath.gyroadd(b, x, k=k), k=k),
        k=k,
    )
    np.testing.assert_allclose(pvmath.gyration(a, b, x, k=k), definition, atol=1e-10)
    a, b = torch.randn(16, 4) * 1e3, torch.randn(16, 4) * 1e3
    np.testing.assert_allclose(
        pvmath.gyration(a, b, x, k=k).norm(dim=-1), x.norm(dim=-1), rtol=1e-8
    )


@pytest.mark.parametrize("distance", [5.0, 20.0, 40.0])
def test_float32_dynamic_range(distance):
    # In float32, distances from the origin stay finite and accurate up to about 44 (where
    # squared norms overflow); geoopt.PoincareBall caps them at about 6.2 (projection eps 4e-3)
    k = torch.tensor(1.0, dtype=torch.float32)
    x = torch.randn(64, 4, dtype=torch.float32)
    t = distance / pvmath.dist0(x, k=k, keepdim=True)
    y = pvmath.gyroscalar(t, x, k=k)
    assert torch.isfinite(y).all()
    np.testing.assert_allclose(pvmath.dist0(y, k=k), distance, rtol=1e-5)
    np.testing.assert_allclose(pvmath.logmap0(y, k=k), t * pvmath.logmap0(x, k=k), rtol=1e-4)


def test_riemannian_optimization(k):
    manifold = geoopt.ProperVelocity(k=k)
    target = torch.randn(4) * 3
    start = torch.randn(4) * 3
    for optim_cls, kwargs in [
        (geoopt.optim.RiemannianSGD, dict(lr=0.2)),
        (geoopt.optim.RiemannianAdam, dict(lr=0.2)),
    ]:
        point = geoopt.ManifoldParameter(start.clone(), manifold=manifold)
        optim = optim_cls([point], **kwargs)
        for _ in range(300):
            optim.zero_grad()
            loss = manifold.dist(point, target) ** 2
            loss.backward()
            optim.step()
        np.testing.assert_allclose(manifold.dist(point, target).item(), 0.0, atol=1e-3)


def test_manifold_api(k):
    manifold = geoopt.ProperVelocity(k=k)
    x = manifold.random_normal(8, 4)
    assert isinstance(x, geoopt.ManifoldTensor)
    manifold.assert_check_point_on_manifold(x)
    np.testing.assert_allclose(manifold.origin(3), torch.zeros(3))
    u = torch.randn(8, 4)
    y = manifold.expmap(x, u)
    np.testing.assert_allclose(manifold.dist(x, y), manifold.norm(x, u), rtol=1e-8)
    np.testing.assert_allclose(manifold.logmap(x, y), u, atol=1e-8)
    with pytest.raises(ValueError):
        geoopt.ProperVelocity(k=-1.0)
    with pytest.raises(ValueError):
        manifold.assert_check_point_on_manifold(torch.tensor([float("nan"), 0.0]))


def test_manifold_methods_delegate_to_math(k):
    manifold = geoopt.ProperVelocity(k=k)
    x, y = torch.randn(8, 4) * 2, torch.randn(8, 4) * 2
    u, v = torch.randn(8, 4), torch.randn(8, 4)
    t = torch.rand(8, 1)
    np.testing.assert_allclose(manifold.logmap0(y), pvmath.logmap0(y, k=k))
    np.testing.assert_allclose(manifold.transp0(y, u), pvmath.parallel_transport0(y, u, k=k))
    np.testing.assert_allclose(
        manifold.transp0back(x, u), pvmath.parallel_transport0back(x, u, k=k)
    )
    np.testing.assert_allclose(manifold.gyration(x, y, u), pvmath.gyration(x, y, u, k=k))
    np.testing.assert_allclose(manifold.geodesic(t, x, y), pvmath.geodesic(t, x, y, k=k))
    np.testing.assert_allclose(
        pvmath.beta(x, k=k), torch.rsqrt(1 + (x * x).sum(-1, keepdim=True) / k)
    )
    z, q = manifold.expmap_transp(x, u, v)
    np.testing.assert_allclose(z, manifold.expmap(x, u))
    np.testing.assert_allclose(q, manifold.transp(x, z, v))
    np.testing.assert_allclose(manifold.transp_follow_expmap(x, u, v), q)
    np.testing.assert_allclose(
        manifold.frechet_variance(x, manifold.frechet_mean(x)),
        pvmath.frechet_variance(x, pvmath.frechet_mean(x, k=k), k=k),
    )


def test_manifold_dtype_device_checks():
    manifold = geoopt.ProperVelocity(k=torch.tensor(1.0, dtype=torch.float64))
    assert manifold.k.dtype == torch.float64
    assert geoopt.ProperVelocity(k=1).k.dtype == torch.get_default_dtype()
    with pytest.raises(ValueError, match="dtype"):
        manifold.random_normal(3, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="device"):
        manifold.random_normal(3, 4, device=torch.device("meta"))
    assert "k=1.0" in repr(manifold)
