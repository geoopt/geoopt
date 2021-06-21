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
    geoopt.manifolds.Stereographic: (3,),
    geoopt.manifolds.SphereProjection: (3,),
    geoopt.manifolds.EuclideanStiefel: (10, 5),
    geoopt.manifolds.BirkhoffPolytope: (4, 4),
    geoopt.manifolds.CanonicalStiefel: (10, 5),
    geoopt.manifolds.Euclidean: (10,),
    geoopt.manifolds.Sphere: (10,),
    geoopt.manifolds.SphereExact: (10,),
    geoopt.manifolds.ProductManifold: (10 + 3 + 6 + 1,),
    geoopt.manifolds.SymmetricPositiveDefinite: (2, 2),
    geoopt.manifolds.UpperHalf: (2, 2),
}


UnaryCase = collections.namedtuple("UnaryCase", "shape,x,ex,v,ev,manifold")


def canonical_stiefel_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.CanonicalStiefel]
    ex = torch.randn(*shape)
    ev = torch.randn(*shape)
    u, _, v = torch.linalg.svd(ex, full_matrices=False)
    x = torch.einsum("...ik,...kj->...ij", u, v)
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
    u, _, v = torch.linalg.svd(ex, full_matrices=False)
    x = torch.einsum("...ik,...kj->...ij", u, v)
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


def proju_original(x, u):
    import geoopt.linalg as linalg

    # takes batch data
    # batch_size, n, _ = x.shape
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-2], x_shape[-1])
    batch_size, n = x.shape[0:2]

    e = torch.ones(batch_size, n, 1, dtype=x.dtype)
    I = torch.unsqueeze(torch.eye(x.shape[-1], dtype=x.dtype), 0).repeat(
        batch_size, 1, 1
    )

    mu = x * u

    A = linalg.block_matrix([[I, x], [torch.transpose(x, 1, 2), I]])

    B = A[:, :, 1:]
    b = torch.cat(
        [
            torch.sum(mu, dim=2, keepdim=True),
            torch.transpose(torch.sum(mu, dim=1, keepdim=True), 1, 2),
        ],
        dim=1,
    )

    zeta = torch.linalg.solve(
        B.transpose(1, 2) @ B, B.transpose(1, 2) @ (b - A[:, :, 0:1])
    )
    alpha = torch.cat(
        [torch.ones(batch_size, 1, 1, dtype=x.dtype), zeta[:, 0 : n - 1]], dim=1
    )
    beta = zeta[:, n - 1 : 2 * n - 1]
    rgrad = mu - (alpha @ e.transpose(1, 2) + e @ beta.transpose(1, 2)) * x

    rgrad = rgrad.reshape(x_shape)
    return rgrad


def birkhoff_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.BirkhoffPolytope]
    ex = torch.randn(*shape, dtype=torch.float64).abs()
    ev = torch.randn(*shape, dtype=torch.float64)
    max_iter = 100
    eps = 1e-12
    tol = 1e-5
    iter = 0
    c = 1.0 / (torch.sum(ex, dim=-2, keepdim=True) + eps)
    r = 1.0 / (torch.matmul(ex, c.transpose(-1, -2)) + eps)
    while iter < max_iter:
        iter += 1
        cinv = torch.matmul(r.transpose(-1, -2), ex)
        if torch.max(torch.abs(cinv * c - 1)) <= tol:
            break
        c = 1.0 / (cinv + eps)
        r = 1.0 / ((ex @ c.transpose(-1, -2)) + eps)
    x = ex * (r @ c)

    v = proju_original(x, ev)
    manifold = geoopt.manifolds.BirkhoffPolytope()
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


def stereographic_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.Stereographic]
    ex = torch.randn(*shape, dtype=torch.float64) / 3
    ev = torch.randn(*shape, dtype=torch.float64) / 3
    x = ex  # default curvature = 0
    ex = x.clone()
    v = ev.clone()
    manifold = geoopt.Stereographic().to(dtype=torch.float64)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.StereographicExact().to(dtype=torch.float64)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def sphere_projection_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.SphereProjection]
    ex = torch.randn(*shape, dtype=torch.float64) / 3
    ev = torch.randn(*shape, dtype=torch.float64) / 3
    x = ex  # default curvature = 0
    ex = x.clone()
    v = ev.clone()
    manifold = geoopt.manifolds.SphereProjection().to(dtype=torch.float64)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case
    manifold = geoopt.manifolds.SphereProjectionExact().to(dtype=torch.float64)
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def sphere_subspace_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.Sphere]
    subspace = torch.rand(shape[-1], 2, dtype=torch.float64)

    Q, _ = geoopt.linalg.batch_linalg.qr(subspace, "reduced")
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

    Q, _ = geoopt.linalg.batch_linalg.qr(complement, "reduced")
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


def spd_case():
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.SymmetricPositiveDefinite]
    ex = torch.randn(*shape, dtype=torch.float64)
    ev = torch.randn(*shape, dtype=torch.float64)
    x = geoopt.linalg.batch_linalg.sym_funcm(
        geoopt.linalg.batch_linalg.sym(ex), torch.abs
    )
    v = geoopt.linalg.batch_linalg.sym(ev)

    manifold = geoopt.SymmetricPositiveDefinite("AIM")
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def upper_half_case():
    # x in Spa, is the result of projecting ex
    # ev is in the tangent space
    # v is the result of projecting ev at x
    torch.manual_seed(42)
    shape = manifold_shapes[geoopt.manifolds.UpperHalf]
    ex = torch.randn(*shape, dtype=torch.complex128)
    ex = geoopt.linalg.batch_linalg.sym(ex)
    x = ex.clone()
    x.imag = geoopt.manifolds.siegel.csym_math.positive_conjugate_projection(x.imag)

    ev = torch.randn(*shape, dtype=torch.complex128) / 10
    ev = geoopt.linalg.batch_linalg.sym(ev)

    real_ev, imag_ev = ev.real, ev.imag
    real_v = x.imag @ real_ev @ x.imag
    imag_v = x.imag @ imag_ev @ x.imag
    v = torch.complex(real_v, imag_v)

    manifold = geoopt.UpperHalf("riem")
    x = geoopt.ManifoldTensor(x, manifold=manifold)
    case = UnaryCase(shape, x, ex, v, ev, manifold)
    yield case


def product_case():
    torch.manual_seed(42)
    ex = [torch.randn(10), torch.randn(3) / 10, torch.randn(3, 2), torch.randn(())]
    ev = [torch.randn(10), torch.randn(3) / 10, torch.randn(3, 2), torch.randn(())]
    manifolds = [
        geoopt.Sphere(),
        geoopt.PoincareBall(),
        geoopt.Stiefel(),
        geoopt.Euclidean(),
    ]
    x = [manifolds[i].projx(ex[i]) for i in range(len(manifolds))]
    v = [manifolds[i].proju(x[i], ev[i]) for i in range(len(manifolds))]

    product_manifold = geoopt.ProductManifold(
        *((manifolds[i], ex[i].shape) for i in range(len(ex)))
    )

    yield UnaryCase(
        manifold_shapes[geoopt.ProductManifold],
        product_manifold.pack_point(*x),
        product_manifold.pack_point(*ex),
        product_manifold.pack_point(*v),
        product_manifold.pack_point(*ev),
        product_manifold,
    )
    # + 1 case without stiefel
    torch.manual_seed(42)
    ex = [torch.randn(10), torch.randn(3) / 10, torch.randn(())]
    ev = [torch.randn(10), torch.randn(3) / 10, torch.randn(())]
    manifolds = [
        geoopt.Sphere(),
        geoopt.PoincareBall(),
        # geoopt.Stiefel(),
        geoopt.Euclidean(),
    ]
    x = [manifolds[i].projx(ex[i]) for i in range(len(manifolds))]
    v = [manifolds[i].proju(x[i], ev[i]) for i in range(len(manifolds))]

    product_manifold = geoopt.ProductManifold(
        *((manifolds[i], ex[i].shape) for i in range(len(ex)))
    )

    yield UnaryCase(
        manifold_shapes[geoopt.ProductManifold],
        product_manifold.pack_point(*x),
        product_manifold.pack_point(*ex),
        product_manifold.pack_point(*v),
        product_manifold.pack_point(*ev),
        product_manifold,
    )


@pytest.fixture(params=[True, False], ids=["Scaled", "NotScaled"])
def scaled(request):
    return request.param


@pytest.fixture(
    params=itertools.chain(
        euclidean_case(),
        sphere_case(),
        sphere_compliment_case(),
        sphere_subspace_case(),
        euclidean_stiefel_case(),
        canonical_stiefel_case(),
        stereographic_case(),
        poincare_case(),
        sphere_projection_case(),
        product_case(),
        birkhoff_case(),
        spd_case(),
        upper_half_case()
    ),
    ids=lambda case: case.manifold.__class__.__name__,
)
def unary_case_base(request):
    return request.param


@pytest.fixture
def unary_case(unary_case_base, scaled):
    if scaled:
        return unary_case_base._replace(
            manifold=geoopt.Scaled(unary_case_base.manifold, 2)
        )
    else:
        return unary_case_base


def test_projection_identity(unary_case):
    x = unary_case.x
    px = unary_case.manifold.projx(x)
    if not isinstance(
        geoopt.utils.canonical_manifold(unary_case.manifold),
        geoopt.manifolds.BirkhoffPolytope,
    ):
        np.testing.assert_allclose(x, px, atol=1e-6)
    else:
        np.testing.assert_allclose(x, px, atol=1e-4)


def test_projection_via_assert(unary_case):
    ex = unary_case.ex
    px = unary_case.manifold.projx(ex)
    unary_case.manifold.assert_check_point_on_manifold(px)
    if not isinstance(
        geoopt.utils.canonical_manifold(unary_case.manifold),
        geoopt.manifolds.BirkhoffPolytope,
    ):
        np.testing.assert_allclose(unary_case.x.detach(), px.detach(), atol=1e-6)
    else:
        np.testing.assert_allclose(unary_case.x.detach(), px.detach(), atol=1e-4)


def test_vector_projection_via_assert(unary_case):
    x = unary_case.x
    ev = unary_case.ev
    v = unary_case.v

    pv = unary_case.manifold.proju(x, ev)

    unary_case.manifold.assert_check_vector_on_tangent(x, pv)

    np.testing.assert_allclose(v.detach(), pv.detach(), atol=1e-6)


def test_broadcast_projx(unary_case):
    X = torch.stack([unary_case.ex] * 4)
    pX = unary_case.manifold.projx(X)
    unary_case.manifold.assert_check_point_on_manifold(pX)
    for px in pX:
        unary_case.manifold.assert_check_point_on_manifold(px)
    for x, px in zip(X, pX):
        pxx = unary_case.manifold.projx(x)
        np.testing.assert_allclose(px.detach(), pxx.detach(), atol=1e-7)


def test_broadcast_proju(unary_case):
    pX = torch.stack([unary_case.x] * 4)
    U = torch.stack([unary_case.v] * 4)
    pU = unary_case.manifold.proju(pX, U)
    unary_case.manifold.assert_check_vector_on_tangent(pX, pU)
    for px, pu in zip(pX, pU):
        unary_case.manifold.assert_check_vector_on_tangent(px, pu, atol=1e-5)
    for px, u, pu in zip(pX, U, pU):
        puu = unary_case.manifold.proju(px, u)
        np.testing.assert_allclose(pu.detach(), puu.detach(), atol=1e-5)


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
        np.testing.assert_allclose(y.detach(), yy.detach(), atol=1e-5)


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
        np.testing.assert_allclose(q.detach(), qq.detach(), atol=1e-5)


def test_reversibility(unary_case):
    if unary_case.manifold.reversible:
        pX = torch.stack([unary_case.x] * 4)
        U = torch.randn(4, *unary_case.shape, dtype=unary_case.x.dtype) / 3
        U = unary_case.manifold.proju(pX, U)
        Z, Q = unary_case.manifold.retr_transp(pX, U, U)
        X1, U1 = unary_case.manifold.retr_transp(Z, -Q, Q)

        np.testing.assert_allclose(X1.detach(), pX.detach(), atol=1e-5)
        np.testing.assert_allclose(U1.detach(), U.detach(), atol=1e-5)
    else:
        pytest.skip(
            "The manifold {} is not supposed to be checked".format(unary_case.manifold)
        )


def test_logmap(unary_case):
    try:
        pX = torch.stack([unary_case.x] * 4)
        U = torch.randn(*unary_case.shape, dtype=unary_case.x.dtype)
        U = unary_case.manifold.proju(pX, U)

        Y = unary_case.manifold.expmap(pX, U)
        Uh = unary_case.manifold.logmap(pX, Y)
        Yh = unary_case.manifold.expmap(pX, Uh)
        np.testing.assert_allclose(Yh.detach(), Y.detach(), atol=1e-6, rtol=1e-6)
        Zero = unary_case.manifold.logmap(pX, pX)
        np.testing.assert_allclose(Zero.detach(), 0.0, atol=1e-6, rtol=1e-6)
    except NotImplementedError:
        pytest.skip("logmap was not implemented for {}".format(unary_case.manifold))


def test_dist(unary_case):
    tangent = unary_case.v
    point = unary_case.x
    tangent_norm = unary_case.manifold.norm(point, tangent)
    try:
        new_point = unary_case.manifold.expmap(point, tangent)
        dist = unary_case.manifold.dist(point, new_point)
        np.testing.assert_allclose(dist.detach(), tangent_norm.detach())
    except NotImplementedError:
        pytest.skip("dist is not implemented for {}".format(unary_case.manifold))
