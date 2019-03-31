import geoopt
import numpy as np
import torch
import collections
import pytest
import functools
import pymanopt.manifolds


@pytest.fixture("module", params=[1, -1])
def retraction_order(request):
    return request.param


# [1] + list(np.random.randn(5))
@pytest.fixture(
    "module", params=[1, -0.27745849, 0.36177604, 0.99467354, -0.34688093, 0.10370687]
)
def t(request):
    return request.param


@pytest.fixture(
    "module",
    params=[
        # match implementation of pymanopt for stiefel
        functools.partial(geoopt.manifolds.Stiefel, canonical=False),
        functools.partial(geoopt.manifolds.Stiefel, canonical=True),
        geoopt.manifolds.PoincareBall,
        geoopt.manifolds.Euclidean,
        geoopt.manifolds.Sphere,
        functools.partial(
            geoopt.manifolds.SphereSubspaceIntersection,
            torch.from_numpy(np.random.RandomState(42).randn(10, 3)),
        ),
        functools.partial(
            geoopt.manifolds.SphereSubspaceComplementIntersection,
            torch.from_numpy(np.random.RandomState(42).randn(10, 3)),
        ),
    ],
)
def manifold(request, retraction_order):
    man = request.param()
    try:
        return man.set_default_order(retraction_order).double()
    except ValueError:
        pytest.skip("not supported retraction order for {}".format(man))


mannopt = {
    geoopt.manifolds.EuclideanStiefel: pymanopt.manifolds.Stiefel,
    geoopt.manifolds.CanonicalStiefel: pymanopt.manifolds.Stiefel,
    geoopt.manifolds.Euclidean: pymanopt.manifolds.Euclidean,
    geoopt.manifolds.Sphere: pymanopt.manifolds.Sphere,
    geoopt.manifolds.SphereSubspaceIntersection: functools.partial(
        pymanopt.manifolds.SphereSubspaceIntersection,
        U=np.random.RandomState(42).randn(10, 3),
    ),
    geoopt.manifolds.SphereSubspaceComplementIntersection: functools.partial(
        pymanopt.manifolds.SphereSubspaceComplementIntersection,
        U=np.random.RandomState(42).randn(10, 3),
    ),
}

# shapes to verify unary element implementation
shapes = {
    geoopt.manifolds.PoincareBall: (3,),
    geoopt.manifolds.EuclideanStiefel: (10, 5),
    geoopt.manifolds.CanonicalStiefel: (10, 5),
    geoopt.manifolds.Euclidean: (1,),
    geoopt.manifolds.Sphere: (10,),
    geoopt.manifolds.SphereSubspaceIntersection: (10,),
    geoopt.manifolds.SphereSubspaceComplementIntersection: (10,),
}

UnaryCase = collections.namedtuple(
    "UnaryCase", "shape,x,ex,v,ev,manifold,manopt_manifold"
)


@pytest.fixture()
def unary_case(manifold):
    shape = shapes[type(manifold)]
    np.random.seed(42)
    torch.manual_seed(43)
    if type(manifold) in mannopt:
        manopt_manifold = mannopt[type(manifold)](*shape)
        rand = manopt_manifold.rand().astype("float64")
        x = geoopt.ManifoldTensor(torch.from_numpy(rand), manifold=manifold)
    else:
        manopt_manifold = None
        x = geoopt.ManifoldTensor(
            torch.randn(shape, dtype=torch.float64) * 0.1, manifold=manifold
        )
    ex = geoopt.ManifoldTensor(torch.randn_like(x), manifold=manifold)
    v = x.proju(torch.randn_like(x))
    ev = torch.randn_like(x)
    return UnaryCase(shape, x, ex, v, ev, manifold, manopt_manifold)


def test_projection_identity(unary_case):
    x = unary_case.x
    xp = unary_case.manifold.projx(x)
    np.testing.assert_allclose(x, xp)


def test_projection_via_assert(unary_case):
    x = unary_case.ex
    xp = unary_case.manifold.projx(x)
    unary_case.manifold.assert_check_point_on_manifold(xp)


def test_vector_projection(unary_case):
    if isinstance(unary_case.manifold, geoopt.manifolds.CanonicalStiefel):
        pytest.skip("pymanopt uses euclidean Stiefel")
    elif unary_case.manopt_manifold is None:
        pytest.skip("pymanopt does not have {}".format(unary_case.manifold))
    x = unary_case.x
    ev = unary_case.ev

    pv = x.proju(ev)
    pv_star = unary_case.manopt_manifold.egrad2rgrad(x.numpy(), ev.numpy())

    np.testing.assert_allclose(pv, pv_star)


def test_vector_projection_via_assert(unary_case):
    x = unary_case.x
    ev = unary_case.ev

    pv = x.proju(ev)

    unary_case.manifold.assert_check_vector_on_tangent(x, pv)


def test_retraction(unary_case, retraction_order, t):
    if unary_case.manopt_manifold is None:
        pytest.skip("pymanopt does not have {}".format(unary_case.manifold))
    if isinstance(unary_case.manifold, geoopt.manifolds.CanonicalStiefel):
        pytest.skip("pymanopt uses euclidean Stiefel")
    x = unary_case.x
    v = unary_case.v

    y = x.retr(v, t=t)
    if retraction_order == 1:
        y_star = unary_case.manopt_manifold.retr(x.numpy(), v.numpy() * t)
        np.testing.assert_allclose(y, y_star)
    elif retraction_order == -1:
        y_star = unary_case.manopt_manifold.exp(x.numpy(), v.numpy() * t)
        np.testing.assert_allclose(y, y_star)


def test_transport(unary_case, t):
    if unary_case.manopt_manifold is None:
        pytest.skip("pymanopt does not have {}".format(unary_case.manifold))
    if isinstance(unary_case.manifold, geoopt.manifolds.CanonicalStiefel):
        pytest.skip("pymanopt uses euclidean Stiefel")
    x = unary_case.x
    v = unary_case.v

    y = x.retr(v, t=t)

    u = x.transp(v, u=v, t=t)

    u_star = unary_case.manopt_manifold.transp(x.numpy(), y.numpy(), v.numpy())

    np.testing.assert_allclose(u, u_star)


def test_broadcast_projx(unary_case):
    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pX = unary_case.manifold.projx(X)
    unary_case.manifold.assert_check_point_on_manifold(pX)
    for px in pX:
        unary_case.manifold.assert_check_point_on_manifold(px)
    for x, px in zip(X, pX):
        pxx = unary_case.manifold.projx(x)
        np.testing.assert_allclose(px, pxx)


def test_broadcast_proju(unary_case):
    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pX = unary_case.manifold.projx(X)
    pU = unary_case.manifold.proju(pX, U)
    unary_case.manifold.assert_check_vector_on_tangent(pX, pU)
    for px, pu in zip(pX, pU):
        unary_case.manifold.assert_check_vector_on_tangent(px, pu, atol=1e-5)
    for px, u, pu in zip(pX, U, pU):
        puu = unary_case.manifold.proju(px, u)
        np.testing.assert_allclose(pu, puu, atol=1e-5)


def test_broadcast_retr(unary_case, t):
    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pX = unary_case.manifold.projx(X)
    pU = unary_case.manifold.proju(pX, U)
    Y = unary_case.manifold.retr(pX, pU, t=t)
    unary_case.manifold.assert_check_point_on_manifold(Y)
    for y in Y:
        unary_case.manifold.assert_check_point_on_manifold(y)
    for px, pu, y in zip(pX, pU, Y):
        yy = unary_case.manifold.retr(px, pu, t=t)
        np.testing.assert_allclose(y, yy, atol=1e-5)


def test_broadcast_transp(unary_case, t):
    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pX = unary_case.manifold.projx(X)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    Y = unary_case.manifold.retr(pX, pU, t=t)
    Q = unary_case.manifold.transp(pX, pV, u=pU, t=t)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    for y, q in zip(Y, Q):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
    for px, pu, pv, y, q in zip(pX, pU, pV, Y, Q):
        qq = unary_case.manifold.transp(px, pv, u=pu, t=t)
        np.testing.assert_allclose(q, qq, atol=1e-5)


def test_broadcast_transp_many(unary_case, t):
    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    F = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pX = unary_case.manifold.projx(X)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    pF = unary_case.manifold.proju(pX, F)
    Y = unary_case.manifold.retr(pX, pU, t=t)
    Q, P = unary_case.manifold.transp(pX, pV, pF, u=pU, t=t)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    unary_case.manifold.assert_check_vector_on_tangent(Y, P)
    for y, q, p in zip(Y, Q, P):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
        unary_case.manifold.assert_check_vector_on_tangent(y, p)
    for px, pu, pv, pf, y, q, p in zip(pX, pU, pV, pF, Y, Q, P):
        qq, pp = unary_case.manifold.transp(px, pv, pf, u=pu, t=t)
        np.testing.assert_allclose(q, qq, atol=1e-5)
        np.testing.assert_allclose(p, pp, atol=1e-5)


def test_broadcast_retr_transp_many(unary_case, t):
    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    V = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    F = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    pX = unary_case.manifold.projx(X)
    pU = unary_case.manifold.proju(pX, U)
    pV = unary_case.manifold.proju(pX, V)
    pF = unary_case.manifold.proju(pX, F)
    Y = unary_case.manifold.retr(pX, pU, t=t)
    Z, Q, P = unary_case.manifold.retr_transp(pX, pV, pF, u=pU, t=t)
    np.testing.assert_allclose(Z, Y, atol=1e-5)
    unary_case.manifold.assert_check_vector_on_tangent(Y, Q)
    unary_case.manifold.assert_check_vector_on_tangent(Y, P)
    for y, q, p in zip(Y, Q, P):
        unary_case.manifold.assert_check_vector_on_tangent(y, q)
        unary_case.manifold.assert_check_vector_on_tangent(y, p)
    for px, pu, pv, pf, y, q, p in zip(pX, pU, pV, pF, Y, Q, P):
        zz, qq, pp = unary_case.manifold.retr_transp(px, pv, pf, u=pu, t=t)
        np.testing.assert_allclose(zz, y, atol=1e-5)
        np.testing.assert_allclose(q, qq, atol=1e-5)
        np.testing.assert_allclose(p, pp, atol=1e-5)


def test_reversibility(unary_case, t):
    if unary_case.manifold.reversible:
        torch.manual_seed(43)
        X = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
        U = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
        X = unary_case.manifold.projx(X)
        U = unary_case.manifold.proju(X, U)
        Z, Q = unary_case.manifold.retr_transp(X, U, u=U, t=t)
        X1, U1 = unary_case.manifold.retr_transp(Z, Q, u=Q, t=-t)

        np.testing.assert_allclose(X1, X, atol=1e-5)
        np.testing.assert_allclose(U1, U, atol=1e-5)
    else:
        pytest.skip("The manifold {} is not supposed to be checked")


def test_dist(unary_case):
    if type(unary_case.manifold)._dist is geoopt.manifolds.base.not_implemented:
        pytest.skip("dist is not implemented for {}".format(unary_case.manifold))
    if unary_case.manopt_manifold is None:
        pytest.skip(
            "dist is not implemented for pymanopt {}".format(unary_case.manifold)
        )
    torch.manual_seed(43)
    x = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
    y = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
    x = unary_case.manifold.projx(x)
    y = unary_case.manifold.projx(y)
    dhat = unary_case.manopt_manifold.dist(x.numpy(), y.numpy())
    d = unary_case.manifold.dist(x, y)
    np.testing.assert_allclose(d, dhat)


def test_logmap(unary_case, t):
    if type(unary_case.manifold)._logmap is geoopt.manifolds.base.not_implemented:
        pytest.skip("logmap is not implemented for {}".format(unary_case.manifold))

    x = unary_case.x
    v = unary_case.v
    if unary_case.manopt_manifold is not None:
        y = unary_case.manopt_manifold.exp(x.numpy(), v.numpy() * t)
        vman = unary_case.manopt_manifold.log(x.numpy(), y)
        vhat = unary_case.manifold.logmap(x, torch.as_tensor(y))
        np.testing.assert_allclose(vhat, vman, atol=1e-7)
    else:
        y = unary_case.manifold.expmap(x, v)
        vhat = unary_case.manifold.logmap(x, torch.as_tensor(y))
    ey = unary_case.manifold.expmap(x, vhat)
    np.testing.assert_allclose(y, ey, atol=1e-7)


def test_logmap_many(unary_case, t):
    if type(unary_case.manifold)._logmap is geoopt.manifolds.base.not_implemented:
        pytest.skip("logmap is not implemented for {}".format(unary_case.manifold))

    torch.manual_seed(43)
    X = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype)
    X = unary_case.manifold.projx(X)
    U = unary_case.manifold.proju(X, U)

    Y = unary_case.manifold.expmap(X, U, t=t)
    Uh = unary_case.manifold.logmap(X, Y)
    Yh = unary_case.manifold.expmap(X, Uh)

    np.testing.assert_allclose(Yh, Y, atol=1e-7)
