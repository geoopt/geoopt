import torch
import random
import numpy as np
import pytest
from geoopt.manifolds import lorentz

@pytest.fixture("function", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture("function", params=[torch.float64, torch.float32])
def dtype(request):
    return request.param


@pytest.fixture
def k(seed, dtype):
    # test broadcasted and non broadcasted versions
    if seed == 30:
        k = torch.tensor(0.0).to(dtype)
    elif seed == 35:
        k = torch.zeros(100, 1, dtype=dtype)
    elif seed > 35:
        k = torch.rand(100, 1, dtype=dtype)
    else:
        k = torch.tensor(random.random()).to(dtype)
    return k + 1e-10


@pytest.fixture
def a(seed, k):
    if seed in {30, 35}:
        a = torch.randn(100, 10, dtype=k.dtype)
    elif seed > 35:
        # do not check numerically unstable regions
        # I've manually observed small differences there
        a = torch.empty(100, 10, dtype=k.dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= (torch.rand_like(k) * k) ** 0.5
    else:
        a = torch.empty(100, 10, dtype=k.dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= random.uniform(0, k) ** 0.5
    return lorentz.math.project(a, k=k)


def test_parallel_transport_a_b(a, b, k):
    # pointing to the center
    v_0 = torch.rand_like(a)
    u_0 = torch.rand_like(a)
    v_1 = lorentz.math.parallel_transport(a, b, v_0, k=k)
    u_1 = loretnz.math.parallel_transport(a, b, u_0, k=k)
    # compute norms
    vu_1 = lorentz.math.inner(b, v_1, u_1, k=k, keepdim=True)
    vu_0 = lorentz.math.inner(a, v_0, u_0, k=k, keepdim=True)
    np.testing.assert_allclose(vu_0, vu_1, atol=1e-6, rtol=1e-6)


def test_expmap_logmap(a, b, k):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    bh = lorentz.math.expmap(x=a, u=lorentz.math.logmap(a, b, k=k), k=k)
    tolerance = {torch.float32: dict(rtol=1e-5, atol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(bh, b, **tolerance[k.dtype])


def test_expmap0_logmap0(a, k):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    v = lorentz.math.logmap0(a, k=k)
    norm = lorentz.math.norm(torch.zeros_like(v), v, k=k, keepdim=True)
    dist = lorentz.math.dist0(a, k=k, keepdim=True)
    bh = lorentz.math.expmap0(v, k=k)
    tolerance = {torch.float32: dict(rtol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(bh, a, **tolerance[k.dtype])
    np.testing.assert_allclose(norm, dist, **tolerance[k.dtype])


def test_geodesic_segement_unit_property(a, b, k):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=k.dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = lorentz.math.geodesic_unit(t, a, b, k=k)
    gamma_ab_t0 = gamma_ab_t[:1]
    gamma_ab_t1 = gamma_ab_t
    dist_ab_t0mt1 = lorentz.math.dist(gamma_ab_t0, gamma_ab_t1, k=k, keepdim=True)
    true_distance_travelled = t.expand_as(dist_ab_t0mt1)
    # we have exactly 12 line segments
    tolerance = {
        torch.float32: dict(atol=1e-6, rtol=1e-5),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(
        dist_ab_t0mt1, true_distance_travelled, **tolerance[k.dtype]
    )


def test_geodesic_segment_length_property(a, b, k):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=k.dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = lorentz.math.geodesic(t, a, b, k=k)
    gamma_ab_t0 = gamma_ab_t[:-1]
    gamma_ab_t1 = gamma_ab_t[1:]
    dist_ab_t0mt1 = lorentz.math.dist(gamma_ab_t0, gamma_ab_t1, k=k, keepdim=True)
    speed = (
        lorentz.math.dist(a, b, k=k, keepdim=True)
        .unsqueeze(0)
        .expand_as(dist_ab_t0mt1)
    )
    # we have exactly 12 line segments
    tolerance = {torch.float32: dict(rtol=1e-5), torch.float64: dict(atol=1e-10)}
    np.testing.assert_allclose(dist_ab_t0mt1, speed / segments, **tolerance[k.dtype])
