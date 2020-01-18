import torch
import random
import numpy as np
import pytest
from geoopt.manifolds import lorentz


@pytest.fixture("module", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture("module", params=[torch.float64, torch.float32])
def dtype(request):
    return request.param


@pytest.fixture
def k(seed, dtype):
    return torch.Tensor([seed - 29.0])


@pytest.fixture
def a(seed, k):
    if seed > 35:
        a = torch.empty(100, 10, dtype=k.dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= (torch.rand_like(k) * k) ** 0.5
    else:
        a = torch.empty(100, 10, dtype=k.dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= random.uniform(0, k) ** 0.5
    return lorentz.math.project(a, k=k)


@pytest.fixture
def b(seed, k):
    if seed > 35:
        b = torch.empty(100, 10, dtype=k.dtype).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= (torch.rand_like(k) * k) ** 0.5
    else:
        b = torch.empty(100, 10, dtype=k.dtype).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= random.uniform(0, k) ** 0.5
    return lorentz.math.project(b, k=k)


def test_expmap_logmap(a, b, k):
    man = lorentz.Lorentz(k=k)
    a = man.projx(a)
    b = man.projx(b)

    bh = man.expmap(a, man.logmap(a, b), project=False)
    tolerance = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(bh, b, **tolerance[k.dtype])


@pytest.mark.xfail
def test_geodesic_segement_unit_property(a, b, k):
    man = lorentz.Lorentz(k=k)
    a = man.projx(a)
    b = man.projx(b)

    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=k.dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = man.geodesic_unit(t, a, b)
    gamma_ab_t0 = gamma_ab_t[:1]
    gamma_ab_t1 = gamma_ab_t
    dist_ab_t0mt1 = man.dist(gamma_ab_t0, gamma_ab_t1, keepdim=True)
    true_distance_travelled = t.expand_as(dist_ab_t0mt1)

    tolerance = {
        torch.float32: dict(atol=1e-6, rtol=1e-5),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(
        dist_ab_t0mt1, true_distance_travelled, **tolerance[k.dtype]
    )


def test_expmap0_logmap0(a, k):
    man = lorentz.Lorentz(k=k)
    a = man.projx(a)
    v = man.logmap0(a)
    norm = man.norm(v, keepdim=True)
    dist = man.dist0(a, keepdim=True)
    bh = man.expmap0(v)
    tolerance = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(bh, a, **tolerance[k.dtype])
    np.testing.assert_allclose(norm, dist, **tolerance[k.dtype])


def test_parallel_transport0_preserves_inner_products(a, k):
    man = lorentz.Lorentz(k=k)
    a = man.projx(a)

    v_0 = torch.rand_like(a) + 1e-5
    u_0 = torch.rand_like(a) + 1e-5

    zero = torch.ones_like(a)
    d = zero.size(1) - 1
    zero = torch.cat(
        (zero.narrow(1, 0, 1) * torch.sqrt(k), zero.narrow(1, 1, d) * 0.0), dim=1
    )

    v_0 = man.proju(zero, v_0)  # project on tangent plane
    u_0 = man.proju(zero, u_0)  # project on tangent plane

    v_a = man.transp0(a, v_0)
    u_a = man.transp0(a, u_0)

    vu_0 = man.inner(v_0, u_0, keepdim=True)
    vu_a = man.inner(v_a, u_a, keepdim=True)
    np.testing.assert_allclose(vu_a, vu_0, atol=1e-5, rtol=1e-5)


def test_parallel_transport0_is_same_as_usual(a, k):
    man = lorentz.Lorentz(k=k)
    a = man.projx(a)
    v_0 = torch.rand_like(a) + 1e-5

    zero = torch.ones_like(a)
    d = zero.size(1) - 1
    zero = torch.cat(
        (zero.narrow(1, 0, 1) * torch.sqrt(k), zero.narrow(1, 1, d) * 0.0), dim=1
    )

    v_a = man.transp0(a, v_0)
    v_a1 = man.transp(zero, a, v_0)
    np.testing.assert_allclose(v_a, v_a1, atol=1e-5, rtol=1e-5)


def test_parallel_transport_a_b(a, b, k):
    man = lorentz.Lorentz(k=k)
    v_0 = torch.rand_like(a)
    u_0 = torch.rand_like(a)

    v_0 = man.proju(a, v_0)  # project on tangent plane
    u_0 = man.proju(a, u_0)  # project on tangent plane

    v_1 = man.transp(a, b, v_0)
    u_1 = man.transp(a, b, u_0)

    vu_1 = man.inner(v_1, u_1, keepdim=True)
    vu_0 = man.inner(v_0, u_0, keepdim=True)

    np.testing.assert_allclose(vu_0, vu_1, atol=1e-5, rtol=1e-5)
