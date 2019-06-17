import geoopt
import numpy as np
import torch
import collections
import pytest


@pytest.fixture("module", params=[1, -1])
def retraction_order(request):
    return request.param


@pytest.fixture("module", params=[1, 2, 3, 4, 5], autouse=True)
def seed(request):
    torch.manual_seed(request.param)
    yield


manifold_shapes = {
    geoopt.manifolds.PoincareBall: (3,),
    geoopt.manifolds.EuclideanStiefel: (10, 5),
    geoopt.manifolds.CanonicalStiefel: (10, 5),
    geoopt.manifolds.Euclidean: (1,),
    geoopt.manifolds.Sphere: (10,),
    geoopt.manifolds.SphereSubspaceIntersection: (10,),
    geoopt.manifolds.SphereSubspaceComplementIntersection: (10,),
}


UnaryCase = collections.namedtuple("UnaryCase", "shape,x,ex,v,ev,manifold")


@pytest.fixture(
    "module",
    params=[
        geoopt.manifolds.EuclideanStiefel,
        geoopt.manifolds.CanonicalStiefel,
        geoopt.manifolds.PoincareBall,
        geoopt.manifolds.Euclidean,
        geoopt.manifolds.Sphere,
        geoopt.manifolds.SphereSubspaceIntersection,
        geoopt.manifolds.SphereSubspaceComplementIntersection,
    ],
)
def unary_case(request, retraction_order):
    Manifold = request.param
    shape = manifold_shapes[Manifold]
    try:
        if issubclass(Manifold, geoopt.manifolds.CanonicalStiefel):
            ex = torch.randn(*shape)
            ev = torch.randn(*shape)
            u, _, v = torch.svd(ex)
            x = u @ v.t()
            v = ev - x @ ev.t() @ x

            manifold = Manifold()
        elif issubclass(Manifold, geoopt.manifolds.EuclideanStiefel):
            ex = torch.randn(*shape)
            ev = torch.randn(*shape)
            u, _, v = torch.svd(ex)
            x = u @ v.t()
            nonsym = x.t() @ ev
            v = ev - x @ (nonsym + nonsym.t()) / 2

            manifold = Manifold()
        elif issubclass(Manifold, geoopt.manifolds.Euclidean):
            ex = torch.randn(*shape)
            ev = torch.randn(*shape)
            x = ex.clone()
            v = ev.clone()
            manifold = Manifold()
        elif issubclass(Manifold, geoopt.manifolds.PoincareBall):
            ex = torch.randn(*shape) / 3
            ev = torch.randn(*shape) / 3
            x = torch.tanh(torch.norm(ex)) * ex / torch.norm(ex)
            ex = x.clone()
            v = ev.clone()

            manifold = Manifold()
        elif issubclass(Manifold, geoopt.SphereSubspaceComplementIntersection):
            complement = torch.rand(shape[-1], 1)

            Q, _ = geoopt.linalg.batch_linalg.qr(complement)
            P = -Q @ Q.transpose(-1, -2)
            P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1

            ex = torch.randn(*shape)
            ev = torch.randn(*shape)
            x = (ex @ P.t()) / torch.norm(ex @ P.t())
            v = (ev - (x @ ev) * x) @ P.t()

            manifold = Manifold(complement)
        elif issubclass(Manifold, geoopt.SphereSubspaceIntersection):
            subspace = torch.rand(shape[-1], 1)

            Q, _ = geoopt.linalg.batch_linalg.qr(subspace)
            P = Q @ Q.t()

            ex = torch.randn(*shape)
            ev = torch.randn(*shape)
            x = (ex @ P.t()) / torch.norm(ex @ P.t())
            v = (ev - (x @ ev) * x) @ P.t()

            manifold = Manifold(subspace)
        elif issubclass(Manifold, geoopt.manifolds.Sphere):
            ex = torch.randn(*shape)
            ev = torch.randn(*shape)
            x = ex / torch.norm(ex)
            v = ev - (x @ ev) * x

            manifold = Manifold()
        else:
            raise NotImplementedError
        x = geoopt.ManifoldTensor(x, manifold=manifold)
        manifold.set_default_order(retraction_order)
        case = UnaryCase(shape, x, ex, v, ev, manifold)
        return case
    except ValueError:
        pytest.skip("not supported retraction order for {}".format(Manifold))


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

    pv = x.proju(ev)

    unary_case.manifold.assert_check_vector_on_tangent(x, pv)

    np.testing.assert_allclose(v, pv, atol=1e-6)


def test_broadcast_projx(unary_case):
    torch.manual_seed(43)
    X = torch.stack([unary_case.ex] * 4)
    pX = unary_case.manifold.projx(X)
    unary_case.manifold.assert_check_point_on_manifold(pX)
    for px in pX:
        unary_case.manifold.assert_check_point_on_manifold(px)
    for x, px in zip(X, pX):
        pxx = unary_case.manifold.projx(x)
        np.testing.assert_allclose(px, pxx, atol=1e-7)


def test_broadcast_proju(unary_case):
    torch.manual_seed(43)
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
    torch.manual_seed(43)
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
    torch.manual_seed(43)
    pX = torch.stack([unary_case.x] * 4)
    U = torch.stack([unary_case.v] * 4)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    Y = unary_case.manifold.retr(pX, pU)
    Q = unary_case.manifold.transp(pX, pV, u=pU)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    for y, q in zip(Y, Q):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
    for px, pu, pv, y, q in zip(pX, pU, pV, Y, Q):
        qq = unary_case.manifold.transp(px, pv, u=pu)
        np.testing.assert_allclose(q, qq, atol=1e-5)


def test_broadcast_transp_many(unary_case):
    torch.manual_seed(43)
    pX = torch.stack([unary_case.x] * 4)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    F = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    pF = unary_case.manifold.proju(pX, F)
    Y = unary_case.manifold.retr(pX, pU)
    Q, P = unary_case.manifold.transp(pX, pV, pF, u=pU)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    unary_case.manifold.assert_check_vector_on_tangent(Y, P)
    for y, q, p in zip(Y, Q, P):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
        unary_case.manifold.assert_check_vector_on_tangent(y, p)
    for px, pu, pv, pf, y, q, p in zip(pX, pU, pV, pF, Y, Q, P):
        qq, pp = unary_case.manifold.transp(px, pv, pf, u=pu)
        np.testing.assert_allclose(q, qq, atol=1e-5)
        np.testing.assert_allclose(p, pp, atol=1e-5)


def test_broadcast_retr_transp_many(unary_case):
    torch.manual_seed(43)
    pX = torch.stack([unary_case.x] * 4)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    F = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    pF = unary_case.manifold.proju(pX, F)
    Y = unary_case.manifold.retr(pX, pU)
    Z, Q, P = unary_case.manifold.retr_transp(pX, pV, pF, u=pU)
    np.testing.assert_allclose(Z, Y, atol=1e-5)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    unary_case.manifold.assert_check_vector_on_tangent(Y, P)
    for y, q, p in zip(Y, Q, P):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
        unary_case.manifold.assert_check_vector_on_tangent(y, p)
    for px, pu, pv, pf, y, q, p in zip(pX, pU, pV, pF, Y, Q, P):
        zz, qq, pp = unary_case.manifold.retr_transp(px, pv, pf, u=pu)
        np.testing.assert_allclose(zz, y, atol=1e-5)
        np.testing.assert_allclose(q, qq, atol=1e-5)
        np.testing.assert_allclose(p, pp, atol=1e-5)


def test_reversibility(unary_case):
    if unary_case.manifold.reversible:
        torch.manual_seed(43)
        pX = torch.stack([unary_case.x] * 4)
        U = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
        U = unary_case.manifold.proju(pX, U)
        Z, Q = unary_case.manifold.retr_transp(pX, U, u=U)
        X1, U1 = unary_case.manifold.retr_transp(Z, Q, u=-Q)

        np.testing.assert_allclose(X1, pX, atol=1e-5)
        np.testing.assert_allclose(U1, U, atol=1e-5)
    else:
        pytest.skip("The manifold {} is not supposed to be checked")


def test_logmap_many(unary_case):
    if type(unary_case.manifold)._logmap is geoopt.manifolds.base.not_implemented:
        pytest.skip("logmap is not implemented for {}".format(unary_case.manifold))

    torch.manual_seed(43)
    pX = torch.stack([unary_case.x] * 4)
    U = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
    U = unary_case.manifold.proju(pX, U)

    Y = unary_case.manifold.expmap(pX, U)
    Uh = unary_case.manifold.logmap(pX, Y)
    Yh = unary_case.manifold.expmap(pX, Uh)

    np.testing.assert_allclose(Yh, Y, atol=1e-6)
