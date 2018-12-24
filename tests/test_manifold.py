import geoopt
import numpy as np
import torch
import collections
import pytest
import pymanopt.manifolds


@pytest.fixture('session', params=[geoopt.manifolds.Stiefel, geoopt.manifolds.Euclidean])
def Manifold(request):
    return request.param


mannopt = {
    geoopt.manifolds.Stiefel: pymanopt.manifolds.Stiefel,
    geoopt.manifolds.Euclidean: pymanopt.manifolds.Euclidean
}

# shapes to verify unary element implementation
shapes = {
    geoopt.manifolds.Stiefel: (10, 5),
    geoopt.manifolds.Euclidean: (1,)
}

UnaryCase = collections.namedtuple('UnaryCase', 'shape,x,ex,v,ev,manifold,manopt_manifold')


@pytest.fixture()
def unary_case(Manifold):
    shape = shapes[Manifold]
    manifold = Manifold()
    manopt_manifold = mannopt[Manifold](*shape)
    np.random.seed(42)
    rand = manopt_manifold.rand()
    x = geoopt.ManifoldTensor(torch.from_numpy(rand), manifold=manifold)
    torch.manual_seed(43)
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


def test_retraction(unary_case):
    x = unary_case.x
    v = unary_case.v

    y = x.retr(v, 1.)
    y_star = unary_case.manopt_manifold.retr(x.numpy(), v.numpy())

    np.testing.assert_allclose(y, y_star)


def test_transport(unary_case):
    x = unary_case.x
    v = unary_case.v

    y = x.retr(v, 1.)

    u = x.transp(v, 1., v)

    u_star = unary_case.manopt_manifold.transp(x.numpy(), y.numpy(), v.numpy())

    np.testing.assert_allclose(u, u_star)

