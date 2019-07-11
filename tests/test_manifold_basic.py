import geoopt
import numpy as np
import torch
import collections
import pytest
import itertools


@pytest.fixture(autouse=True, params=[1, 2, 3, 4, 5])
def seed(request):
    torch.manual_seed(request.param)
    yield


@pytest.fixture(autouse=True, params=[torch.float64], ids=lambda t: str(t))
def use_floatX(request):
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    yield request.param
    torch.set_default_dtype(dtype_old)


manifold_shapes = {
    geoopt.manifolds.PoincareBall: (3,),
    geoopt.manifolds.EuclideanStiefel: (10, 5),
    geoopt.manifolds.CanonicalStiefel: (10, 5),
    geoopt.manifolds.Euclidean: (10,),
    geoopt.manifolds.Sphere: (10,),
    geoopt.manifolds.SphereExact: (10,),
}


UnaryCase = collections.namedtuple("UnaryCase", "shape,x,ex,v,ev,manifold")


def canonical_stiefel_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.CanonicalStiefel]
    ex = torch.randn(*shape)
    ev = torch.randn(*shape)
    u, _, v = torch.svd(ex)
    x = u @ v.t()
    v = ev - x @ ev.t() @ x
    manifold = geoopt.manifolds.CanonicalStiefel()
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def euclidean_stiefel_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.EuclideanStiefel]
    ex = torch.randn(*shape, dtype=torch.float64)
    ev = torch.randn(*shape, dtype=torch.float64)
    u, _, v = torch.svd(ex)
    x = u @ v.t()
    nonsym = x.t() @ ev
    v = ev - x @ (nonsym + nonsym.t()) / 2

    manifold = geoopt.manifolds.EuclideanStiefel()
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.manifolds.EuclideanStiefelExact()
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def euclidean_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.Euclidean]
    ex = torch.randn(*shape, dtype=torch.float64)
    ev = torch.randn(*shape, dtype=torch.float64)
    x = ex.clone()
    v = ev.clone()
    manifold = geoopt.Euclidean(ndim=1)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def poincare_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.PoincareBall]
    ex = torch.randn(*shape, dtype=torch.float64) / 3
    ev = torch.randn(*shape, dtype=torch.float64) / 3
    x = torch.tanh(torch.norm(ex)) * ex / torch.norm(ex)
    ex = x.clone()
    v = ev.clone()
    manifold = geoopt.PoincareBall().to(dtype=torch.float64)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.PoincareBallExact().to(dtype=torch.float64)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def sphere_subspace_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.Sphere]
    subspace = torch.rand(shape[-1], 2, dtype=torch.float64)

    Q, _ = geoopt.linalg.batch_linalg.qr(subspace)
    P = Q @ Q.t()

    ex = torch.randn(*shape, dtype=torch.float64)
    ev = torch.randn(*shape, dtype=torch.float64)
    x = (ex @ P.t()) / torch.norm(ex @ P.t())
    v = (ev - (x @ ev) * x) @ P.t()

    manifold = geoopt.Sphere(intersection=subspace)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.SphereExact(intersection=subspace)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def sphere_compliment_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.Sphere]
    complement = torch.rand(shape[-1], 1, dtype=torch.float64)

    Q, _ = geoopt.linalg.batch_linalg.qr(complement)
    P = -Q @ Q.transpose(-1, -2)
    P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1

    ex = torch.randn(*shape, dtype=torch.float64)
    ev = torch.randn(*shape, dtype=torch.float64)
    x = (ex @ P.t()) / torch.norm(ex @ P.t())
    v = (ev - (x @ ev) * x) @ P.t()

    manifold = geoopt.Sphere(complement=complement)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.SphereExact(complement=complement)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def sphere_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.Sphere]
    ex = torch.randn(*shape, dtype=torch.float64)
    ev = torch.randn(*shape, dtype=torch.float64)
    x = ex / torch.norm(ex)
    v = ev - (x @ ev) * x

    manifold = geoopt.Sphere()
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.SphereExact()
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


@pytest.fixture(
    "module",
    params=itertools.chain(
        euclidean_case(),
        sphere_case(),
        sphere_compliment_case(),
        sphere_subspace_case(),
        euclidean_stiefel_case(),
        canonical_stiefel_case(),
        poincare_case(),
    ),
    ids=lambda case: case.manifold.__class__.__name__,
)
def unary_case(request):
    return request.param


def test_projection_identity(unary_case):
    x = unary_case.x
    px = unary_case.manifold.projx(x)
    np.testing.assert_allclose(x, px, atol=1e-6)


def test_projection_via_assert(unary_case):
    ex = unary_case.ex
    px = unary_case.manifold.projx(ex)
    unary_case.manifold.assert_check_point_on_manifold(px)
    np.testing.assert_allclose(unary_case.x, px, atol=1e-6)


def test_vector_projection_via_assert(unary_case):
    x = unary_case.x
    ev = unary_case.ev
    v = unary_case.v

    pv = unary_case.manifold.proju(x, ev)

    unary_case.manifold.assert_check_vector_on_tangent(x, pv)

    np.testing.assert_allclose(v, pv, atol=1e-6)


def test_broadcast_projx(unary_case):
    X = torch.stack([unary_case.ex] * 4)
    pX = unary_case.manifold.projx(X)
    unary_case.manifold.assert_check_point_on_manifold(pX)
    for px in pX:
        unary_case.manifold.assert_check_point_on_manifold(px)
    for x, px in zip(X, pX):
        pxx = unary_case.manifold.projx(x)
        np.testing.assert_allclose(px, pxx, atol=1e-7)


def test_broadcast_proju(unary_case):
    pX = torch.stack([unary_case.x] * 4)
    U = torch.stack([unary_case.v] * 4)
    pU = unary_case.manifold.proju(pX, U)
    unary_case.manifold.assert_check_vector_on_tangent(pX, pU)
    for px, pu in zip(pX, pU):
        unary_case.manifold.assert_check_vector_on_tangent(px, pu, atol=1e-5)
    for px, u, pu in zip(pX, U, pU):
        puu = unary_case.manifold.proju(px, u)
        np.testing.assert_allclose(pu, puu, atol=1e-5)


def test_broadcast_retr(unary_case):
    pX = torch.stack([unary_case.x] * 4)
    U = torch.stack([unary_case.v] * 4)
    pU = unary_case.manifold.proju(pX, U)
    Y = unary_case.manifold.retr(pX, pU)
    unary_case.manifold.assert_check_point_on_manifold(Y)
    for y in Y:
        unary_case.manifold.assert_check_point_on_manifold(y)
    for px, pu, y in zip(pX, pU, Y):
        yy = unary_case.manifold.retr(px, pu)
        np.testing.assert_allclose(y, yy, atol=1e-5)


def test_broadcast_transp(unary_case):
    pX = torch.stack([unary_case.x] * 4)
    U = torch.stack([unary_case.v] * 4)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    Y = unary_case.manifold.retr(pX, pU)
    Q = unary_case.manifold.transp_follow_retr(pX, pU, pV)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    for y, q in zip(Y, Q):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
    for px, pu, pv, y, q in zip(pX, pU, pV, Y, Q):
        qq = unary_case.manifold.transp_follow_retr(px, pu, pv)
        np.testing.assert_allclose(q, qq, atol=1e-5)


def test_broadcast_transp_many(unary_case):
    pX = torch.stack([unary_case.x] * 4)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    F = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    pF = unary_case.manifold.proju(pX, F)
    Y = unary_case.manifold.retr(pX, pU)
    Q, P = unary_case.manifold.transp_follow_retr(pX, pU, pV, pF)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    unary_case.manifold.assert_check_vector_on_tangent(Y, P)
    for y, q, p in zip(Y, Q, P):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
        unary_case.manifold.assert_check_vector_on_tangent(y, p)
    for px, pu, pv, pf, y, q, p in zip(pX, pU, pV, pF, Y, Q, P):
        qq, pp = unary_case.manifold.transp_follow_retr(px, pu, pv, pf)
        np.testing.assert_allclose(q, qq, atol=1e-5)
        np.testing.assert_allclose(p, pp, atol=1e-5)


def test_broadcast_retr_transp_many(unary_case):
    pX = torch.stack([unary_case.x] * 4)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    F = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    pF = unary_case.manifold.proju(pX, F)
    Y = unary_case.manifold.retr(pX, pU)
    Z, Q, P = unary_case.manifold.retr_transp(pX, pU, pV, pF)
    np.testing.assert_allclose(Z, Y, atol=1e-5)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    unary_case.manifold.assert_check_vector_on_tangent(Y, P)
    for y, q, p in zip(Y, Q, P):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
        unary_case.manifold.assert_check_vector_on_tangent(y, p)
    for px, pu, pv, pf, y, q, p in zip(pX, pU, pV, pF, Y, Q, P):
        zz, qq, pp = unary_case.manifold.retr_transp(px, pu, pv, pf)
        np.testing.assert_allclose(zz, y, atol=1e-5)
        np.testing.assert_allclose(q, qq, atol=1e-5)
        np.testing.assert_allclose(p, pp, atol=1e-5)


def test_reversibility(unary_case):
    if unary_case.manifold.reversible:
        pX = torch.stack([unary_case.x] * 4)
        U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype) / 3
        U = unary_case.manifold.proju(pX, U)
        Z, Q = unary_case.manifold.retr_transp(pX, U, U)
        X1, U1 = unary_case.manifold.retr_transp(Z, -Q, Q)

        np.testing.assert_allclose(X1, pX, atol=1e-5)
        np.testing.assert_allclose(U1, U, atol=1e-5)
    else:
        pytest.skip(
            "The manifold {} is not supposed to be checked".format(unary_case.manifold)
        )


def test_logmap_many(unary_case):
    try:
        pX = torch.stack([unary_case.x] * 4)
        U = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
        U = unary_case.manifold.proju(pX, U)

        Y = unary_case.manifold.expmap(pX, U)
        Uh = unary_case.manifold.logmap(pX, Y)
        Yh = unary_case.manifold.expmap(pX, Uh)

        np.testing.assert_allclose(Yh, Y, atol=1e-6, rtol=1e-6)
    except NotImplementedError:
        pytest.skip("logmap was not implemented for {}".format(unary_case.manifold))
