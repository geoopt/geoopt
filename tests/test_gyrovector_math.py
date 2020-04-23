"""
Tests ideas are taken mostly from https://github.com/dalab/hyperbolic_nn/blob/master/util.py with some changes
"""
import torch
import random
import numpy as np
import pytest
import warnings
import itertools
from geoopt.manifolds import stereographic


@pytest.fixture("function", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(
    "function", params=[torch.float64, torch.float32], ids=["float64", "float32"]
)
def dtype(request):
    return request.param


def tolerant_allclose_check(a, b, strict=True, **tolerance):
    if strict:
        np.testing.assert_allclose(a.detach(), b.detach(), **tolerance)
    else:
        try:
            np.testing.assert_allclose(a.detach(), b.detach(), **tolerance)
        except AssertionError as e:
            assert not torch.isnan(a).any(), "Found nans"
            assert not torch.isnan(b).any(), "Found nans"
            warnings.warn("Unstable numerics: " + " | ".join(str(e).splitlines()[3:6]))


@pytest.fixture(params=[True, False], ids=["negative", "positive"])
def negative(request):
    return request.param


@pytest.fixture()
def strict(seed, dtype, negative):
    return seed in {30, 31} and dtype == torch.float64 or negative


# c = -k
@pytest.fixture
def c(seed, dtype, negative):
    # test broadcasted and non broadcasted versions
    if seed == 30:  # strict seed
        c = torch.tensor(0.0).to(dtype)
    elif seed == 31:  # strict seed too
        c = torch.tensor(1.0).to(dtype)
    elif seed == 39:
        c = 10 ** torch.arange(-15, 1, dtype=dtype)[:, None]
    elif seed == 35:
        c = torch.zeros(100, 1, dtype=dtype)
    elif seed > 35:
        c = torch.rand(100, 1, dtype=dtype)
    else:
        c = torch.tensor(random.random()).to(dtype)
    if not negative:
        c = -c
    return c.requires_grad_(True)


@pytest.fixture
def k(c):
    return -c


@pytest.fixture
def manifold(k):
    return stereographic.Stereographic(k=k, learnable=True)


@pytest.fixture
def B(c):
    if c.dim() > 1:
        return c.shape[0]
    else:
        return 100


@pytest.fixture
def a(seed, c, manifold, B, dtype):
    r = manifold.radius
    a = torch.empty(B, 10, dtype=dtype).normal_(-1, 1)
    a /= a.norm(dim=-1, keepdim=True)
    a *= torch.where(torch.isfinite(r), r, torch.ones((), dtype=dtype)).clamp_max_(100)
    a *= torch.rand_like(a)
    return manifold.projx(a).detach().requires_grad_(True)


@pytest.fixture
def b(seed, c, manifold, B, dtype):
    r = manifold.radius
    a = torch.empty(B, 10, dtype=dtype).normal_(-1, 1)
    a /= a.norm(dim=-1, keepdim=True)
    a *= torch.where(torch.isfinite(r), r, torch.ones((), dtype=dtype)).clamp_max_(100)
    a *= torch.rand_like(a)
    return manifold.projx(a).detach().requires_grad_(True)


@pytest.fixture
def logunif_input(dtype):
    inp = 10 ** torch.arange(-15, 1, dtype=dtype)
    inp = torch.cat([-inp.flip(0), torch.zeros([1], dtype=dtype), inp])
    return inp.requires_grad_(True)


def test_tanh_grad(logunif_input):
    stereographic.math.tanh(logunif_input).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()


def test_artanh_grad(logunif_input):
    stereographic.math.artanh(logunif_input).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()


def test_arsinh_grad(logunif_input):
    stereographic.math.arsinh(logunif_input).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()


def test_tan_k_grad(logunif_input):
    k = logunif_input.detach().clone().requires_grad_()
    stereographic.math.tan_k(logunif_input[None], k[:, None]).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()
    assert torch.isfinite(k.grad).all()


def test_artan_k_grad(logunif_input):
    k = logunif_input.detach().clone().requires_grad_()
    stereographic.math.artan_k(logunif_input[None], k[:, None]).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()
    assert torch.isfinite(k.grad).all()


def test_arsin_k_grad(logunif_input):
    k = logunif_input.detach().clone().requires_grad_()
    stereographic.math.arsin_k(logunif_input[None], k[:, None]).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()
    assert torch.isfinite(k.grad).all()


def test_sin_k_grad(logunif_input):
    k = logunif_input.detach().clone().requires_grad_()
    stereographic.math.sin_k(logunif_input[None], k[:, None]).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()
    assert torch.isfinite(k.grad).all()


def test_project_k_grad(logunif_input):
    vec = logunif_input[:, None] * torch.ones(logunif_input.shape[0], 10)
    k = logunif_input.detach().clone().requires_grad_()
    stereographic.math.project(vec, k=k[:, None]).sum().backward()
    assert torch.isfinite(logunif_input.grad).all()
    assert torch.isfinite(k.grad).all()


def test_mobius_addition_left_cancelation(a, b, manifold, dtype):
    res = manifold.mobius_add(-a, manifold.mobius_add(a, b))
    tolerance = {torch.float32: dict(atol=5e-5, rtol=5e-4), torch.float64: dict()}
    np.testing.assert_allclose(res.detach(), b.detach(), **tolerance[dtype])
    res.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_mobius_addition_zero_a(b, manifold):
    a = torch.zeros_like(b)
    res = manifold.mobius_add(a, b)
    np.testing.assert_allclose(res.detach(), b.detach())
    res.sum().backward()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_mobius_addition_zero_b(a, c, manifold):
    b = torch.zeros_like(a)
    res = manifold.mobius_add(a, b)
    np.testing.assert_allclose(res.detach(), a.detach())
    res.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_mobius_addition_negative_cancellation(a, manifold, dtype):
    res = manifold.mobius_add(a, -a)
    tolerance = {
        torch.float32: dict(atol=1e-4, rtol=1e-6),
        torch.float64: dict(atol=1e-6),
    }
    np.testing.assert_allclose(res.detach(), torch.zeros_like(res), **tolerance[dtype])
    res.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_mobius_negative_addition(a, b, manifold, dtype):
    res = manifold.mobius_add(-b, -a)
    res1 = -manifold.mobius_add(b, a)
    tolerance = {
        torch.float32: dict(atol=1e-7, rtol=1e-6),
        torch.float64: dict(atol=1e-10),
    }

    np.testing.assert_allclose(res.detach(), res1.detach(), **tolerance[dtype])
    res.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


@pytest.mark.parametrize("n", list(range(5)))
def test_n_additions_via_scalar_multiplication(n, a, dtype, negative, manifold, strict):
    n = torch.as_tensor(n, dtype=a.dtype).requires_grad_()
    y = torch.zeros_like(a)
    for _ in range(int(n.item())):
        y = manifold.mobius_add(a, y)
    ny = manifold.mobius_scalar_mul(n, a)
    if negative:
        tolerance = {
            torch.float32: dict(atol=4e-5, rtol=1e-3),
            torch.float64: dict(atol=1e-5, rtol=1e-3),
        }
    else:
        tolerance = {
            torch.float32: dict(atol=2e-6, rtol=1e-3),
            torch.float64: dict(atol=1e-5, rtol=1e-3),
        }
    tolerant_allclose_check(y, ny, strict=strict, **tolerance[dtype])
    ny.sum().backward()
    assert torch.isfinite(n.grad).all()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


@pytest.fixture
def r1(seed, dtype, B):
    if seed % 3 == 0:
        return (
            torch.tensor(random.uniform(-1, 1), dtype=dtype)
            .detach()
            .requires_grad_(True)
        )
    else:
        return (torch.rand(B, 1, dtype=dtype) * 2 - 1).detach().requires_grad_(True)


@pytest.fixture
def r2(seed, dtype, B):
    if seed % 3 == 1:
        return (
            torch.tensor(random.uniform(-1, 1), dtype=dtype)
            .detach()
            .requires_grad_(True)
        )
    else:
        return (torch.rand(B, 1, dtype=dtype) * 2 - 1).detach().requires_grad_(True)


def test_scalar_multiplication_distributive(a, r1, r2, manifold, dtype):
    res = manifold.mobius_scalar_mul(r1 + r2, a)
    res1 = manifold.mobius_add(
        manifold.mobius_scalar_mul(r1, a), manifold.mobius_scalar_mul(r2, a),
    )
    res2 = manifold.mobius_add(
        manifold.mobius_scalar_mul(r1, a), manifold.mobius_scalar_mul(r2, a),
    )
    tolerance = {
        torch.float32: dict(atol=5e-6, rtol=1e-4),
        torch.float64: dict(atol=1e-7, rtol=1e-4),
    }
    np.testing.assert_allclose(res1.detach(), res.detach(), **tolerance[dtype])
    np.testing.assert_allclose(res2.detach(), res.detach(), **tolerance[dtype])
    res.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(r1.grad).all()
    assert torch.isfinite(r2.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_scalar_multiplication_associative(a, r1, r2, manifold, dtype):
    res = manifold.mobius_scalar_mul(r1 * r2, a)
    res1 = manifold.mobius_scalar_mul(r1, manifold.mobius_scalar_mul(r2, a))
    res2 = manifold.mobius_scalar_mul(r2, manifold.mobius_scalar_mul(r1, a))
    tolerance = {
        torch.float32: dict(atol=1e-5, rtol=1e-5),
        torch.float64: dict(atol=1e-7, rtol=1e-7),
    }
    np.testing.assert_allclose(res1.detach(), res.detach(), **tolerance[dtype])
    np.testing.assert_allclose(res2.detach(), res.detach(), **tolerance[dtype])
    res.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(r1.grad).all()
    assert torch.isfinite(r2.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_scaling_property(a, r1, manifold, dtype):
    x1 = a / a.norm(dim=-1, keepdim=True)
    ra = manifold.mobius_scalar_mul(r1, a)
    x2 = manifold.mobius_scalar_mul(abs(r1), a) / ra.norm(dim=-1, keepdim=True)
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=1e-6),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(x1.detach(), x2.detach(), **tolerance[dtype])
    x2.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(r1.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_geodesic_borders(a, b, manifold, dtype):
    geo0 = manifold.geodesic(torch.tensor(0.0, dtype=dtype), a, b)
    geo1 = manifold.geodesic(torch.tensor(1.0, dtype=dtype), a, b)
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=5e-5),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(geo0.detach(), a.detach(), **tolerance[dtype])
    np.testing.assert_allclose(geo1.detach(), b.detach(), **tolerance[dtype])
    (geo0 + geo1).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_geodesic_segment_length_property(a, b, manifold, dtype):
    extra_dims = len(a.shape)
    segments = 12
    t = torch.linspace(0, 1, segments + 1, dtype=dtype).view(
        (segments + 1,) + (1,) * extra_dims
    )
    gamma_ab_t = manifold.geodesic(t, a, b)
    gamma_ab_t0 = gamma_ab_t[:-1]
    gamma_ab_t1 = gamma_ab_t[1:]
    dist_ab_t0mt1 = manifold.dist(gamma_ab_t0, gamma_ab_t1, keepdim=True)
    speed = manifold.dist(a, b, keepdim=True).unsqueeze(0).expand_as(dist_ab_t0mt1)
    # we have exactly 12 line segments
    tolerance = {
        torch.float32: dict(rtol=1e-5, atol=5e-3),
        torch.float64: dict(rtol=1e-5, atol=5e-3),
    }
    length = speed / segments
    np.testing.assert_allclose(
        dist_ab_t0mt1.detach(), length.detach(), **tolerance[dtype]
    )
    (length + dist_ab_t0mt1).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_geodesic_segement_unit_property(a, b, manifold, dtype):
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
        torch.float32: dict(atol=2e-4, rtol=5e-5),
        torch.float64: dict(atol=1e-10),
    }
    np.testing.assert_allclose(
        dist_ab_t0mt1.detach(), true_distance_travelled.detach(), **tolerance[dtype]
    )
    (true_distance_travelled + dist_ab_t0mt1).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_expmap_logmap(a, b, manifold, dtype):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    bh = manifold.expmap(x=a, u=manifold.logmap(a, b))
    tolerance = {torch.float32: dict(rtol=1e-5, atol=5e-5), torch.float64: dict()}
    np.testing.assert_allclose(bh.detach(), b.detach(), **tolerance[dtype])
    bh.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_expmap0_logmap0(a, manifold, dtype):
    # this test appears to be numerical unstable once a and b may appear on the opposite sides
    v = manifold.logmap0(a)
    norm = manifold.norm(torch.zeros_like(v), v, keepdim=True)
    dist = manifold.dist0(a, keepdim=True)
    bh = manifold.expmap0(v)
    tolerance = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict()}
    np.testing.assert_allclose(bh.detach(), a.detach(), **tolerance[dtype])
    np.testing.assert_allclose(norm.detach(), dist.detach(), **tolerance[dtype])
    (bh.sum() + dist.sum()).backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_matvec_zeros(a, manifold):
    mat = a.new_zeros((3, a.shape[-1]))
    z = manifold.mobius_matvec(mat, a)
    np.testing.assert_allclose(z.detach(), 0.0)
    z.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_matvec_via_equiv_fn_apply(a, negative, manifold, strict, dtype):
    mat = a.new(3, a.shape[-1]).normal_()
    y = manifold.mobius_fn_apply(lambda x: x @ mat.transpose(-1, -2), a)
    y1 = manifold.mobius_matvec(mat, a)
    tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict()}

    tolerant_allclose_check(y, y1, strict=strict, **tolerance[dtype])
    y.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_mobiusify(a, c, negative, strict, dtype):
    mat = a.new(3, a.shape[-1]).normal_()

    @stereographic.math.mobiusify
    def matvec(x):
        return x @ mat.transpose(-1, -2)

    y = matvec(a, k=-c)
    y1 = stereographic.math.mobius_matvec(mat, a, k=-c)
    tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict()}

    tolerant_allclose_check(y, y1, strict=strict, **tolerance[dtype])
    y.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(c.grad).all()


def test_matvec_chain_via_equiv_fn_apply(a, negative, manifold, dtype):
    mat1 = a.new(a.shape[-1], a.shape[-1]).normal_()
    mat2 = a.new(a.shape[-1], a.shape[-1]).normal_()
    y = manifold.mobius_fn_apply_chain(
        a, lambda x: x @ mat1.transpose(-1, -2), lambda x: x @ mat2.transpose(-1, -2),
    )
    y1 = manifold.mobius_matvec(mat1, a)
    y1 = manifold.mobius_matvec(mat2, y1)
    tolerance = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float64: dict()}

    tolerant_allclose_check(y, y1, strict=negative, **tolerance[dtype])
    y.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_transp0_preserves_inner_products(a, manifold):
    # pointing to the center
    v_0 = torch.rand_like(a) + 1e-5
    u_0 = torch.rand_like(a) + 1e-5
    zero = torch.zeros_like(a)
    v_a = manifold.transp0(a, v_0)
    u_a = manifold.transp0(a, u_0)
    # compute norms
    vu_0 = manifold.inner(zero, v_0, u_0, keepdim=True)
    vu_a = manifold.inner(a, v_a, u_a, keepdim=True)
    np.testing.assert_allclose(vu_a.detach(), vu_0.detach(), atol=1e-6, rtol=1e-6)
    (vu_0 + vu_a).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_transp0_is_same_as_usual(a, manifold):
    # pointing to the center
    v_0 = torch.rand_like(a) + 1e-5
    zero = torch.zeros_like(a)
    v_a = manifold.transp0(a, v_0)
    v_a1 = manifold.transp(zero, a, v_0)
    # compute norms
    np.testing.assert_allclose(v_a.detach(), v_a1.detach(), atol=1e-6, rtol=1e-6)
    (v_a + v_a1).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_transp_a_b(a, b, manifold):
    # pointing to the center
    v_0 = torch.rand_like(a)
    u_0 = torch.rand_like(a)
    v_1 = manifold.transp(a, b, v_0)
    u_1 = manifold.transp(a, b, u_0)
    # compute norms
    vu_1 = manifold.inner(b, v_1, u_1, keepdim=True)
    vu_0 = manifold.inner(a, v_0, u_0, keepdim=True)
    np.testing.assert_allclose(vu_0.detach(), vu_1.detach(), atol=1e-6, rtol=1e-6)
    (vu_0 + vu_1).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_add_infinity_and_beyond(a, b, c, negative, manifold, dtype):
    _a = a
    if torch.isclose(c, c.new_zeros(())).any():
        pytest.skip("zero not checked")
    infty = b * 10000000
    for i in range(100):
        z = manifold.expmap(a, infty, project=False)
        z = manifold.projx(z)
        assert not torch.isnan(z).any(), ("Found nans", i, z)
        assert torch.isfinite(z).all(), ("Found Infs", i, z)
        z = manifold.mobius_scalar_mul(
            torch.tensor(1000.0, dtype=z.dtype), z, project=False
        )
        z = manifold.projx(z)
        assert not torch.isnan(z).any(), ("Found nans", i, z)
        assert torch.isfinite(z).all(), ("Found Infs", i, z)

        infty = manifold.transp(a, z, infty)
        assert torch.isfinite(infty).all(), (i, infty)
        a = z
    z = manifold.expmap(a, -infty)
    # they just need to be very far, exact answer is not supposed
    tolerance = {
        torch.float32: dict(rtol=3e-1, atol=2e-1),
        torch.float64: dict(rtol=1e-1, atol=1e-3),
    }
    if negative:
        np.testing.assert_allclose(z.detach(), -a.detach(), **tolerance[dtype])
    else:
        assert not torch.isnan(z).any(), "Found nans"
        assert not torch.isnan(a).any(), "Found nans"


def test_mobius_coadd(a, b, negative, manifold, strict):
    # (a \boxplus_c b) \ominus_c b = a
    ah = manifold.mobius_sub(manifold.mobius_coadd(a, b), b)
    tolerant_allclose_check(a, ah, strict=strict, atol=5e-5)
    ah.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_mobius_cosub(a, b, negative, manifold, strict):
    # (a \oplus_c b) \boxminus b = a
    ah = manifold.mobius_cosub(manifold.mobius_add(a, b), b)
    tolerant_allclose_check(a, ah, strict=strict, atol=1e-5)
    ah.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_distance2plane(a, manifold):
    v = torch.rand_like(a).requires_grad_()
    vr = v / manifold.norm(a, v, keepdim=True)
    z = manifold.expmap(a, vr)
    dist1 = manifold.dist(a, z)
    dist = manifold.dist2plane(z, a, vr)

    np.testing.assert_allclose(dist.detach(), dist1.detach(), atol=2e-4, rtol=1e-4)
    (dist + dist1).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(v.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_sproj(manifold, a):
    ma = manifold.sproj(manifold.inv_sproj(a))
    np.testing.assert_allclose(ma.detach(), a.detach(), atol=1e-5)
    ma.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


def test_antipode(manifold, negative, a, dtype, seed):
    if seed == 39:
        pytest.skip("This is amazingly unstable when tested against extreme values")
    ma = manifold.antipode(a)
    if manifold.k.le(0).all():
        np.testing.assert_allclose(ma.detach(), -a.detach())
    else:
        s = manifold.inv_sproj(a)
        ms = manifold.inv_sproj(ma)
        tolerance = {torch.float32: dict(atol=1e-5), torch.float64: dict(atol=1e-6)}
        np.testing.assert_allclose(ms.detach(), -s.detach(), **tolerance[dtype])
        ma.sum().backward()
        assert torch.isfinite(a.grad).all()
        assert torch.isfinite(manifold.k.grad).all()


@pytest.mark.parametrize("_k,lincomb", itertools.product([-1, 0, 1], [True, False]))
def test_weighted_midpoint(_k, lincomb):
    manifold = stereographic.Stereographic(_k, learnable=True)
    a = manifold.random(2, 3, 10).requires_grad_(True)
    mid = manifold.weighted_midpoint(a, lincomb=lincomb)
    assert torch.isfinite(mid).all()
    assert mid.shape == (a.shape[-1],)
    mid.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert not torch.isclose(manifold.k.grad, manifold.k.new_zeros(()))


@pytest.mark.parametrize("_k,lincomb", itertools.product([-1, 0, 1], [True, False]))
def test_weighted_midpoint_reduce_dim(_k, lincomb):
    manifold = stereographic.Stereographic(_k, learnable=True)
    a = manifold.random(2, 3, 10).requires_grad_(True)
    mid = manifold.weighted_midpoint(a, reducedim=[0], lincomb=lincomb)
    assert mid.shape == a.shape[-2:]
    assert torch.isfinite(mid).all()
    mid.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert not torch.isclose(manifold.k.grad, manifold.k.new_zeros(()))


@pytest.mark.parametrize("_k,lincomb", itertools.product([-1, 0, 1], [True, False]))
def test_weighted_midpoint_weighted(_k, lincomb):
    manifold = stereographic.Stereographic(_k, learnable=True)
    a = manifold.random(2, 3, 10).requires_grad_(True)
    mid = manifold.weighted_midpoint(
        a, reducedim=[0], lincomb=lincomb, weights=torch.rand_like(a[..., 0])
    )
    assert mid.shape == a.shape[-2:]
    assert torch.isfinite(mid).all()
    mid.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert not torch.isclose(manifold.k.grad, manifold.k.new_zeros(()))


@pytest.mark.parametrize("_k,lincomb", itertools.product([-1, 0, 1], [True, False]))
def test_weighted_midpoint_zero(_k, lincomb):
    manifold = stereographic.Stereographic(_k, learnable=True)
    a = manifold.random(2, 3, 10).requires_grad_(True)
    mid = manifold.weighted_midpoint(
        a, reducedim=[0], lincomb=lincomb, weights=torch.zeros_like(a[..., 0])
    )
    assert mid.shape == a.shape[-2:]
    assert torch.allclose(mid, torch.zeros_like(mid))
    mid.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(manifold.k.grad).all()


@pytest.mark.parametrize("lincomb", [True, False])
def test_weighted_midpoint_euclidean(lincomb):
    manifold = stereographic.Stereographic(0)
    a = manifold.random(2, 3, 10).requires_grad_(True)
    mid = manifold.weighted_midpoint(a, reducedim=[0], lincomb=lincomb)
    assert mid.shape == a.shape[-2:]
    if lincomb:
        assert torch.allclose(mid, a.sum(0))
    else:
        assert torch.allclose(mid, a.mean(0))


@pytest.mark.parametrize("_k,lincomb", itertools.product([-1, 0, 1], [True, False]))
def test_weighted_midpoint_weighted_zero_sum(_k, lincomb):
    manifold = stereographic.Stereographic(_k, learnable=True)
    a = manifold.expmap0(torch.eye(3, 10)).detach().requires_grad_(True)
    weights = torch.rand_like(a[..., 0])
    weights = weights - weights.sum() / weights.numel()
    mid = manifold.weighted_midpoint(a, lincomb=lincomb, weights=weights)
    if _k == 0 and lincomb:
        np.testing.assert_allclose(
            mid.detach(),
            torch.cat([weights, torch.zeros(a.size(-1) - a.size(0))]),
            atol=1e-6,
        )
    assert mid.shape == a.shape[-1:]
    assert torch.isfinite(mid).all()
    mid.sum().backward()
    assert torch.isfinite(a.grad).all()
