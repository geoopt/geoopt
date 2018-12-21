import geoopt
import numpy.testing
import torch
import pytest


def test_stiefel_2d():
    tens1 = geoopt.ManifoldTensor(20, 10, manifold=geoopt.Stiefel()).normal_().proj_()
    vect1 = tens1.proju(torch.randn(*tens1.shape))
    newt = tens1.retr(vect1, 1.0)
    tens1.manifold.assert_check_point_on_manifold(newt)


def test_stiefel_3d():
    tens1 = (
        geoopt.ManifoldTensor(2, 20, 10, manifold=geoopt.Stiefel()).normal_().proj_()
    )
    vect1 = tens1.proju(torch.randn(*tens1.shape))
    t = torch.randn(tens1.shape[0])
    newt = tens1.retr(vect1, t)
    newt_manual = list()
    newt_manual.append(tens1.manifold.retr(tens1[0], vect1[0], t[0]))
    newt_manual.append(tens1.manifold.retr(tens1[1], vect1[1], t[1]))
    newt_manual = torch.stack(newt_manual)
    numpy.testing.assert_allclose(newt_manual, newt, atol=1e-5)
    tens1.manifold.assert_check_point_on_manifold(newt)


@pytest.mark.parametrize("Manifold", [geoopt.Stiefel, geoopt.Euclidean])
def test_reversible_retraction(Manifold):
    man = Manifold()
    x = torch.randn((10,) * man.ndim)
    x.set_(man.projx(x))
    t = 0.1
    u = torch.randn_like(x)
    u.set_(man.proju(x, u))
    xt, ut = man.retr_transp(x, u, t, u)
    xtmt = man.retr(xt, ut, -t)
    numpy.testing.assert_allclose(x, xtmt, atol=1e-5)


@pytest.mark.parametrize("Manifold", [geoopt.Stiefel, geoopt.Euclidean])
def test_transp_many(Manifold):
    man = Manifold()
    x = torch.randn((10,) * man.ndim)
    x.set_(man.projx(x))
    t = 0.1
    u = torch.randn_like(x)
    u.set_(man.proju(x, u))
    vs = [man.proju(x, torch.randn_like(u)) for _ in range(3)]
    qvs1 = man.transp(x, u, t, *vs)
    qvs2 = tuple(man.transp(x, u, t, v) for v in vs)
    for qv1, qv2 in zip(qvs1, qvs2):
        numpy.testing.assert_allclose(qv1, qv2, atol=1e-5)


@pytest.mark.parametrize("Manifold", [geoopt.Stiefel, geoopt.Euclidean])
def test_transp_many(Manifold):
    man = Manifold()
    x = torch.randn((10,) * man.ndim)
    x.set_(man.projx(x))
    t = 0.1
    u = torch.randn_like(x)
    u.set_(man.proju(x, u))
    vs = [man.proju(x, torch.randn_like(u)) for _ in range(3)]
    qvs1 = man.retr_transp(x, u, t, *vs)
    qvs2 = tuple(man.transp(x, u, t, v) for v in vs)
    qvs2 = (man.retr(x, u, t),) + qvs2

    for qv1, qv2 in zip(qvs1, qvs2):
        numpy.testing.assert_allclose(qv1, qv2, atol=1e-5)
