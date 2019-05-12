"""
Tests ideas are taken mostly from https://github.com/dalab/hyperbolic_nn/blob/master/util.py with some changes
"""
import torch
import random
import numpy as np
import pytest
from geoopt.manifolds import poincare


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
def c(seed, dtype):
    # test broadcasted and non broadcasted versions
    if seed == 30:
        c = torch.tensor(0.0).to(dtype)
    elif seed == 35:
        c = torch.zeros(100, 1, dtype=dtype)
    elif seed > 35:
        c = torch.rand(100, 1, dtype=dtype)
    else:
        c = torch.tensor(random.random()).to(dtype)
    return c + 1e-10


@pytest.fixture
def a(seed, c):
    if seed in {30, 35}:
        a = torch.randn(100, 10, dtype=c.dtype)
    elif seed > 35:
        # do not check numerically unstable regions
        # I've manually observed small differences there
        a = torch.empty(100, 10, dtype=c.dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= (torch.rand_like(c) * c) ** 0.5
    else:
        a = torch.empty(100, 10, dtype=c.dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= random.uniform(0, c) ** 0.5
    return poincare.math.project(a, c=c)


@pytest.fixture
def b(seed, c):
    if seed in {30, 35}:
        b = torch.randn(100, 10, dtype=c.dtype)
    elif seed > 35:
        b = torch.empty(100, 10, dtype=c.dtype).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= (torch.rand_like(c) * c) ** 0.5
    else:
        b = torch.empty(100, 10, dtype=c.dtype).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= random.uniform(0, c) ** 0.5
    return poincare.math.project(b, c=c)


def test_mobius_addition_left_cancelation(a, b, c):
    res = poincare.math.mobius_add(-a, poincare.math.mobius_add(a, b, c=c), c=c)
    tolerance = {torch.float32: dict(atol=1e-6, rtol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(res, b, **tolerance[c.dtype])


def test_mobius_addition_zero_a(b, c):
    a = torch.zeros(100, 10, dtype=c.dtype)
    res = poincare.math.mobius_add(a, b, c=c)
    np.testing.assert_allclose(res, b)


def test_mobius_addition_zero_b(a, c):
    b = torch.zeros(100, 10, dtype=c.dtype)
    res = poincare.math.mobius_add(a, b, c=c)
    np.testing.assert_allclose(res, a)


def test_mobius_addition_negative_cancellation(a, c):
    res = poincare.math.mobius_add(a, -a, c=c)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(res, torch.zeros_like(res), **tolerance[c.dtype])


def test_mobius_negative_addition(a, b, c):
    res = poincare.math.mobius_add(-b, -a, c=c)
    res1 = -poincare.math.mobius_add(b, a, c=c)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(res, res1, **tolerance[c.dtype])


@pytest.mark.parametrize("n", list(range(5)))
def test_n_additions_via_scalar_multiplication(n, a, c):
    y = torch.zeros_like(a)
    for _ in range(n):
        y = poincare.math.mobius_add(a, y, c=c)
    ny = poincare.math.mobius_scalar_mul(n, a, c=c)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(y, ny, **tolerance[c.dtype])


@pytest.fixture
def r1(seed, dtype):
    if seed % 3 == 0:
        return random.uniform(-1, 1)
    else:
        return torch.rand(100, 1, dtype=dtype) * 2 - 1


@pytest.fixture
def r2(seed, dtype):
    if seed % 3 == 1:
        return random.uniform(-1, 1)
    else:
        return torch.rand(100, 1, dtype=dtype) * 2 - 1


def test_scalar_multiplication_distributive(a, c, r1, r2):
    res = poincare.math.mobius_scalar_mul(r1 + r2, a, c=c)
    res1 = poincare.math.mobius_add(
        poincare.math.mobius_scalar_mul(r1, a, c=c),
        poincare.math.mobius_scalar_mul(r2, a, c=c),
        c=c,
    )
    res2 = poincare.math.mobius_add(
        poincare.math.mobius_scalar_mul(r1, a, c=c),
        poincare.math.mobius_scalar_mul(r2, a, c=c),
        c=c,
    )
    tolerance = {
        torch.float32: dict(atol=1e-6, rtol=1e-7),
        torch.float64: dict(atol=1e-7, rtol=1e-10),
    }
    np.testing.assert_allclose(res1, res, **tolerance[c.dtype])
    np.testing.assert_allclose(res2, res, **tolerance[c.dtype])


def test_scalar_multiplication_associative(a, c, r1, r2):
    res = poincare.math.mobius_scalar_mul(r1 * r2, a, c=c)
    res1 = poincare.math.mobius_scalar_mul(
        r1, poincare.math.mobius_scalar_mul(r2, a, c=c), c=c
    )
    res2 = poincare.math.mobius_scalar_mul(
        r2, poincare.math.mobius_scalar_mul(r1, a, c=c), c=c
    )
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),  # worked with rtol=1e-7 locally
        torch.float64: dict(atol=1e-7, rtol=1e-10),
    }
    np.testing.assert_allclose(res1, res, **tolerance[c.dtype])
    np.testing.assert_allclose(res2, res, **tolerance[c.dtype])


def test_scaling_property(a, c, r1):
    x1 = a / a.norm(dim=-1, keepdim=True)
    ra = poincare.math.mobius_scalar_mul(r1, a, c=c)
    x2 = poincare.math.mobius_scalar_mul(abs(r1), a, c=c) / ra.norm(
        dim=-1, keepdim=True
    )
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(x1, x2, **tolerance[c.dtype])


def test_geodesic_borders(a, b, c):
    geo0 = poincare.math.geodesic(0.0, a, b, c=c)
    geo1 = poincare.math.geodesic(1.0, a, b, c=c)
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(geo0, a, **tolerance[c.dtype])
    np.testing.assert_allclose(geo1, b, **tolerance[c.dtype])


def test_geodesic_segment_length_property(a, b, c):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=c.dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = poincare.math.geodesic(t, a, b, c=c)
    gamma_ab_t0 = gamma_ab_t[:-1]
    gamma_ab_t1 = gamma_ab_t[1:]
    dist_ab_t0mt1 = poincare.math.dist(gamma_ab_t0, gamma_ab_t1, c=c, keepdim=True)
    speed = (
        poincare.math.dist(a, b, c=c, keepdim=True)
        .unsqueeze(0)
        .expand_as(dist_ab_t0mt1)
    )
    # we have exactly 12 line segments
    tolerance = {torch.float32: dict(rtol=1e-5), torch.float64: dict(atol=1e-10)}
    np.testing.assert_allclose(dist_ab_t0mt1, speed / segments, **tolerance[c.dtype])


def test_geodesic_segement_unit_property(a, b, c):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=c.dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = poincare.math.geodesic_unit(t, a, b, c=c)
    gamma_ab_t0 = gamma_ab_t[:1]
    gamma_ab_t1 = gamma_ab_t
    dist_ab_t0mt1 = poincare.math.dist(gamma_ab_t0, gamma_ab_t1, c=c, keepdim=True)
    true_distance_travelled = t.expand_as(dist_ab_t0mt1)
    # we have exactly 12 line segments
    tolerance = {
        torch.float32: dict(atol=1e-6, rtol=1e-5),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(
        dist_ab_t0mt1, true_distance_travelled, **tolerance[c.dtype]
    )


def test_expmap_logmap(a, b, c):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    bh = poincare.math.expmap(x=a, u=poincare.math.logmap(a, b, c=c), c=c)
    tolerance = {torch.float32: dict(rtol=1e-5, atol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(bh, b, **tolerance[c.dtype])


def test_expmap0_logmap0(a, c):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    v = poincare.math.logmap0(a, c=c)
    norm = poincare.math.norm(torch.zeros_like(v), v, c=c, keepdim=True)
    dist = poincare.math.dist0(a, c=c, keepdim=True)
    bh = poincare.math.expmap0(v, c=c)
    tolerance = {torch.float32: dict(rtol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(bh, a, **tolerance[c.dtype])
    np.testing.assert_allclose(norm, dist, **tolerance[c.dtype])


def test_matvec_zeros(a, c):
    mat = a.new_zeros(3, a.shape[-1])
    z = poincare.math.mobius_matvec(mat, a, c=c)
    np.testing.assert_allclose(z, 0.0)


def test_matvec_via_equiv_fn_apply(a, c):
    mat = a.new(3, a.shape[-1]).normal_()
    y = poincare.math.mobius_fn_apply(lambda x: x @ mat.transpose(-1, -2), a, c=c)
    y1 = poincare.math.mobius_matvec(mat, a, c=c)
    tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(y, y1, **tolerance[c.dtype])


def test_mobiusify(a, c):
    mat = a.new(3, a.shape[-1]).normal_()

    @poincare.math.mobiusify
    def matvec(x):
        return x @ mat.transpose(-1, -2)

    y = matvec(a, c=c)
    y1 = poincare.math.mobius_matvec(mat, a, c=c)
    tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(y, y1, **tolerance[c.dtype])


def test_matvec_chain_via_equiv_fn_apply(a, c):
    mat1 = a.new(a.shape[-1], a.shape[-1]).normal_()
    mat2 = a.new(a.shape[-1], a.shape[-1]).normal_()
    y = poincare.math.mobius_fn_apply_chain(
        a,
        lambda x: x @ mat1.transpose(-1, -2),
        lambda x: x @ mat2.transpose(-1, -2),
        c=c,
    )
    y1 = poincare.math.mobius_matvec(mat1, a, c=c)
    y1 = poincare.math.mobius_matvec(mat2, y1, c=c)
    np.testing.assert_allclose(y, y1, atol=1e-5)


def test_parallel_transport0_preserves_inner_products(a, c):
    # pointing to the center
    v_0 = torch.rand_like(a) + 1e-5
    u_0 = torch.rand_like(a) + 1e-5
    zero = torch.zeros_like(a)
    v_a = poincare.math.parallel_transport0(a, v_0, c=c)
    u_a = poincare.math.parallel_transport0(a, u_0, c=c)
    # compute norms
    vu_0 = poincare.math.inner(zero, v_0, u_0, c=c, keepdim=True)
    vu_a = poincare.math.inner(a, v_a, u_a, c=c, keepdim=True)
    np.testing.assert_allclose(vu_a, vu_0, atol=1e-6, rtol=1e-6)


def test_parallel_transport0_is_same_as_usual(a, c):
    # pointing to the center
    v_0 = torch.rand_like(a) + 1e-5
    zero = torch.zeros_like(a)
    v_a = poincare.math.parallel_transport0(a, v_0, c=c)
    v_a1 = poincare.math.parallel_transport(zero, a, v_0, c=c)
    # compute norms
    np.testing.assert_allclose(v_a, v_a1, atol=1e-6, rtol=1e-6)


def test_parallel_transport_a_b(a, b, c):
    # pointing to the center
    v_0 = torch.rand_like(a)
    u_0 = torch.rand_like(a)
    v_1 = poincare.math.parallel_transport(a, b, v_0, c=c)
    u_1 = poincare.math.parallel_transport(a, b, u_0, c=c)
    # compute norms
    vu_1 = poincare.math.inner(b, v_1, u_1, c=c, keepdim=True)
    vu_0 = poincare.math.inner(a, v_0, u_0, c=c, keepdim=True)
    np.testing.assert_allclose(vu_0, vu_1, atol=1e-6, rtol=1e-6)


def test_add_infinity_and_beyond(a, b, c):
    infty = b * 10000000
    for i in range(100):
        z = poincare.math.expmap(a, infty, c=c)
        z = poincare.math.project(z, c=c)
        z = poincare.math.mobius_scalar_mul(1000.0, z, c=c)
        z = poincare.math.project(z, c=c)
        infty = poincare.math.parallel_transport(a, z, infty, c=c)
        assert np.isfinite(z).all(), (i, z)
        assert np.isfinite(infty).all(), (i, infty)
        a = z
    z = poincare.math.expmap(a, -infty, c=c)
    # they just need to be very far, exact answer is not supposed
    tolerance = {
        torch.float32: dict(rtol=3e-1, atol=2e-1),
        torch.float64: dict(rtol=1e-1, atol=1e-3),
    }
    np.testing.assert_allclose(z, -a, **tolerance[c.dtype])


def test_mobius_coadd(a, b, c):
    # (a \boxplus_c b) \ominus_c b = a
    ah = poincare.math.mobius_sub(poincare.math.mobius_coadd(a, b, c=c), b, c=c)
    np.testing.assert_allclose(ah, a, atol=1e-5)


def test_mobius_cosub(a, b, c):
    # (a \oplus_c b) \boxminus b = a
    ah = poincare.math.mobius_cosub(poincare.math.mobius_add(a, b, c=c), b, c=c)
    np.testing.assert_allclose(ah, a, atol=1e-5)


def test_distance2plane(a, c):
    v = torch.rand_like(a)
    vr = v / poincare.math.norm(a, v, c=c, keepdim=True)
    z = poincare.math.expmap(a, vr, c=c)
    dist1 = poincare.math.dist(a, z, c=c)
    dist = poincare.math.dist2plane(z, a, vr, c=c)

    np.testing.assert_allclose(dist, dist1, atol=1e-5, rtol=1e-5)
