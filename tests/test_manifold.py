import geoopt
import numpy.testing
import torch


def test_stiefel_2d():
    tens1 = geoopt.ManifoldTensor(10, 20, manifold=geoopt.Stiefel()).normal_().proj_()
    vect1 = tens1.proju(torch.randn(*tens1.shape))
    newt = tens1.retr(vect1, 1.)
    assert isinstance(newt, geoopt.ManifoldTensor)
    numpy.testing.assert_allclose(newt, newt.manifold.projx(newt), atol=1e-5)


def test_stiefel_3d():
    tens1 = geoopt.ManifoldTensor(2, 10, 20, manifold=geoopt.Stiefel()).normal_().proj_()
    vect1 = tens1.proju(torch.randn(*tens1.shape))
    t = torch.randn(tens1.shape[0])
    newt = tens1.retr(vect1, t)
    newt_manual = list()
    newt_manual.append(tens1.manifold.retr(tens1[0], vect1[0], t[0]))
    newt_manual.append(tens1.manifold.retr(tens1[1], vect1[1], t[1]))
    newt_manual = torch.stack(newt_manual)
    assert isinstance(newt, geoopt.ManifoldTensor)
    numpy.testing.assert_allclose(newt_manual, newt, atol=1e-5)
    numpy.testing.assert_allclose(newt, newt.manifold.projx(newt), atol=1e-5)
