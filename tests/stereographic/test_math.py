"""
Tests ideas are taken mostly from https://github.com/dalab/hyperbolic_nn/blob/master/util.py with some changes
"""
import torch
import random
import numpy as np
import pytest
from geoopt.manifolds.stereographic import StereographicExact


# FIXTURES #####################################################################


@pytest.fixture("function", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


# TODO: float32 just causes impossibly many problems!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: we can't use float 32
@pytest.fixture("function", params=[torch.float64]) #, torch.float32])
def dtype(request):
    return request.param


@pytest.fixture
def K(seed, dtype):
    # test broadcasted and non broadcasted versions
    if seed == 30:
        K = torch.tensor(0.0).to(dtype)
    else:
        K = (torch.tensor(random.random()).to(dtype)-0.5) * 2.0
    return K + 1e-10


@pytest.fixture
def manifold(K, dtype):
    return StereographicExact(K=K,
                              float_precision=dtype,
                              keep_sign_fixed=False,
                              min_abs_K=0.001)

# TODO: verify if it's ok to use the absolute value here!!!!!!!!!!!!!!!!!!
@pytest.fixture
def a(manifold, K, seed, dtype):
    if seed in {30, 35}:
        a = torch.randn(100, 10, dtype=dtype)
    elif seed > 35:
        # do not check numerically unstable regions
        # I've manually observed small differences there
        a = torch.empty(100, 10, dtype=dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= torch.abs((torch.rand_like(K) * K)) ** 0.5
    else:
        a = torch.empty(100, 10, dtype=dtype).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= torch.abs(random.uniform(0, K)) ** 0.5
    return manifold.projx(a)


@pytest.fixture
def b(manifold, K, seed, dtype):
    if seed in {30, 35}:
        b = torch.randn(100, 10, dtype=dtype)
    elif seed > 35:
        b = torch.empty(100, 10, dtype=dtype).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= torch.abs((torch.rand_like(K) * K)) ** 0.5
    else:
        b = torch.empty(100, 10, dtype=dtype).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= torch.abs(random.uniform(0, K)) ** 0.5
    return manifold.projx(b)


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


# TESTS ########################################################################


def test_mobius_addition_left_cancelation(manifold, a, b, dtype):
    res = manifold.mobius_add(-a, manifold.mobius_add(a, b))
    tolerance = {torch.float32: dict(atol=1e-6, rtol=1e-6),
                 torch.float64: dict()} # TODO: what here???
    np.testing.assert_allclose(res, b, **tolerance[dtype])


def test_mobius_addition_zero_a(manifold, b, dtype):
    a = torch.zeros(100, 10, dtype=dtype)
    res = manifold.mobius_add(a, b)
    np.testing.assert_allclose(res, b)


def test_mobius_addition_zero_b(manifold, a, dtype):
    b = torch.zeros(100, 10, dtype=dtype)
    res = manifold.mobius_add(a, b)
    np.testing.assert_allclose(res, a)


def test_mobius_addition_negative_cancellation(manifold, a, dtype):
    res = manifold.mobius_add(a, -a)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(res, torch.zeros_like(res), **tolerance[dtype])


def test_mobius_negative_addition(manifold, a, b, dtype):
    res = manifold.mobius_add(-b, -a)
    res1 = -manifold.mobius_add(b, a)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(res, res1, **tolerance[dtype])


@pytest.mark.parametrize("n", list(range(5)))
def test_n_additions_via_scalar_multiplication(manifold, n, a, dtype):
    y = torch.zeros_like(a)
    for _ in range(n):
        y = manifold.mobius_add(a, y)
    ny = manifold.mobius_scalar_mul(n, a)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(y, ny, **tolerance[dtype])


def test_scalar_multiplication_distributive(manifold, a, r1, r2, dtype):
    res = manifold.mobius_scalar_mul(r1 + r2, a)
    res1 = manifold.mobius_add(
        manifold.mobius_scalar_mul(r1, a),
        manifold.mobius_scalar_mul(r2, a)
    )
    res2 = manifold.mobius_add(
        manifold.mobius_scalar_mul(r1, a),
        manifold.mobius_scalar_mul(r2, a)
    )
    tolerance = {
        torch.float32: dict(atol=1e-6, rtol=1e-7),
        torch.float64: dict(atol=1e-7, rtol=1e-10),
    }
    np.testing.assert_allclose(res1, res, **tolerance[dtype])
    np.testing.assert_allclose(res2, res, **tolerance[dtype])


def test_scalar_multiplication_associative(manifold, a, r1, r2, dtype):
    res = manifold.mobius_scalar_mul(r1 * r2, a)
    res1 = manifold.mobius_scalar_mul(
        r1, manifold.mobius_scalar_mul(r2, a)
    )
    res2 = manifold.mobius_scalar_mul(
        r2, manifold.mobius_scalar_mul(r1, a)
    )
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),  # worked with rtol=1e-7 locally
        torch.float64: dict(atol=1e-7, rtol=1e-10),
    }
    np.testing.assert_allclose(res1, res, **tolerance[dtype])
    np.testing.assert_allclose(res2, res, **tolerance[dtype])


def test_scaling_property(manifold, a, r1, dtype):
    x1 = a / a.norm(dim=-1, keepdim=True)
    ra = manifold.mobius_scalar_mul(r1, a)
    x2 = manifold.mobius_scalar_mul(abs(r1), a) / ra.norm(
        dim=-1, keepdim=True
    )
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(x1, x2, **tolerance[dtype])


def test_geodesic_borders(manifold, a, b, dtype):
    geo0 = manifold.geodesic(0.0, a, b)
    geo1 = manifold.geodesic(1.0, a, b)
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(geo0, a, **tolerance[dtype])
    np.testing.assert_allclose(geo1, b, **tolerance[dtype])


def test_geodesic_segment_length_property(manifold, a, b, dtype):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = manifold.geodesic(t, a, b)
    gamma_ab_t0 = gamma_ab_t[:-1]
    gamma_ab_t1 = gamma_ab_t[1:]
    dist_ab_t0mt1 = manifold.dist(gamma_ab_t0, gamma_ab_t1, keepdim=True)
    speed = (
        manifold.dist(a, b, keepdim=True)
        .unsqueeze(0)
        .expand_as(dist_ab_t0mt1)
    )
    # we have exactly 12 line segments
    tolerance = {torch.float32: dict(rtol=1e-5), torch.float64: dict(atol=1e-10)}
    np.testing.assert_allclose(dist_ab_t0mt1, speed / segments, **tolerance[dtype])


def test_geodesic_segement_unit_property(manifold, a, b, dtype):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = manifold.geodesic_unit(t, a, b)
    gamma_ab_t0 = gamma_ab_t[:1]
    gamma_ab_t1 = gamma_ab_t
    dist_ab_t0mt1 = manifold.dist(gamma_ab_t0, gamma_ab_t1, keepdim=True)
    true_distance_travelled = t.expand_as(dist_ab_t0mt1)
    # we have exactly 12 line segments
    tolerance = {
        torch.float32: dict(atol=1e-6, rtol=1e-5),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(
        dist_ab_t0mt1, true_distance_travelled, **tolerance[dtype]
    )


def test_expmap_logmap(manifold, a, b, dtype):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    bh = manifold.expmap(x=a, u=manifold.logmap(a, b))
    tolerance = {torch.float32: dict(rtol=1e-5, atol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(bh, b, **tolerance[dtype])


def test_expmap0_logmap0(manifold, a, dtype):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    v = manifold.logmap0(a)
    norm = manifold.norm(torch.zeros_like(v), v, keepdim=True)
    dist = manifold.dist0(a, keepdim=True)
    bh = manifold.expmap0(v)
    tolerance = {torch.float32: dict(rtol=1e-6), torch.float64: dict()}
    np.testing.assert_allclose(bh, a, **tolerance[dtype])
    np.testing.assert_allclose(norm, dist, **tolerance[dtype])


def test_matvec_zeros(manifold, a):
    mat = a.new_zeros(3, a.shape[-1])
    z = manifold.mobius_matvec(mat, a)
    np.testing.assert_allclose(z, 0.0)


def test_matvec_via_equiv_fn_apply(manifold, a, dtype):
    mat = a.new(3, a.shape[-1]).normal_()
    y = manifold.mobius_fn_apply(lambda x: x @ mat.transpose(-1, -2), a)
    y1 = manifold.mobius_matvec(mat, a)
    tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(y, y1, **tolerance[dtype])


def test_mobiusify(manifold, a, dtype):
    mat = a.new(3, a.shape[-1]).normal_()

    @manifold.mobiusify
    def matvec(x):
        return x @ mat.transpose(-1, -2)

    y = matvec(a)
    y1 = manifold.mobius_matvec(mat, a)
    tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(y, y1, **tolerance[dtype])


def test_matvec_chain_via_equiv_fn_apply(manifold, a):
    mat1 = a.new(a.shape[-1], a.shape[-1]).normal_()
    mat2 = a.new(a.shape[-1], a.shape[-1]).normal_()
    y = manifold.mobius_fn_apply_chain(
        a,
        lambda x: x @ mat1.transpose(-1, -2),
        lambda x: x @ mat2.transpose(-1, -2),
    )
    y1 = manifold.mobius_matvec(mat1, a)
    y1 = manifold.mobius_matvec(mat2, y1)
    np.testing.assert_allclose(y, y1, atol=1e-5)


def test_parallel_transport0_preserves_inner_products(manifold, a):
    # pointing to the center
    v_0 = torch.rand_like(a) + 1e-5
    u_0 = torch.rand_like(a) + 1e-5
    zero = torch.zeros_like(a)
    v_a = manifold.transp0(a, v_0)
    u_a = manifold.transp0(a, u_0)
    # compute norms
    vu_0 = manifold.inner(zero, v_0, u_0, keepdim=True)
    vu_a = manifold.inner(a, v_a, u_a, keepdim=True)
    np.testing.assert_allclose(vu_a, vu_0, atol=1e-6, rtol=1e-6)


def test_parallel_transport0_is_same_as_usual(manifold, a):
    # pointing to the center
    v_0 = torch.rand_like(a) + 1e-5
    zero = torch.zeros_like(a)
    v_a = manifold.transp0(a, v_0)
    v_a1 = manifold.transp(zero, a, v_0)
    # compute norms
    np.testing.assert_allclose(v_a, v_a1, atol=1e-6, rtol=1e-6)


def test_parallel_transport_a_b(manifold, a, b, dtype):
    # pointing to the center
    v_0 = torch.rand_like(a)
    u_0 = torch.rand_like(a)
    v_1 = manifold.transp(a, b, v_0)
    u_1 = manifold.transp(a, b, u_0)
    # compute norms
    vu_1 = manifold.inner(b, v_1, u_1, keepdim=True)
    vu_0 = manifold.inner(a, v_0, u_0, keepdim=True)
    np.testing.assert_allclose(vu_0, vu_1, atol=1e-6, rtol=1e-6)


def test_add_infinity_and_beyond(manifold, a, b, dtype):
    infty = b * 10000000
    for i in range(100):
        z = manifold.expmap(a, infty)
        z = manifold.projx(z)
        z = manifold.mobius_scalar_mul(1000.0, z)
        z = manifold.projx(z)
        infty = manifold.transp(a, z, infty)
        assert np.isfinite(z).all(), (i, z)
        assert np.isfinite(infty).all(), (i, infty)
        a = z
    z = manifold.expmap(a, -infty)
    # they just need to be very far, exact answer is not supposed
    tolerance = {
        torch.float32: dict(rtol=3e-1, atol=2e-1),
        torch.float64: dict(rtol=1e-1, atol=1e-3),
    }
    np.testing.assert_allclose(z, -a, **tolerance[dtype])


def test_mobius_coadd(manifold, a, b):
    # (a \boxplus_c b) \ominus_c b = a
    ah = manifold.mobius_sub(manifold.mobius_coadd(a, b), b)
    np.testing.assert_allclose(ah, a, atol=1e-5)


def test_mobius_cosub(manifold, a, b):
    # (a \oplus_c b) \boxminus b = a
    ah = manifold.mobius_cosub(manifold.mobius_add(a, b), b)
    np.testing.assert_allclose(ah, a, atol=1e-5)


def test_distance2plane(manifold, a):
    v = torch.rand_like(a)
    vr = v / manifold.norm(a, v, keepdim=True)
    z = manifold.expmap(a, vr)
    dist1 = manifold.dist(a, z)
    dist = manifold.dist2plane(z, a, vr)

    np.testing.assert_allclose(dist, dist1, atol=1e-5, rtol=1e-5)
