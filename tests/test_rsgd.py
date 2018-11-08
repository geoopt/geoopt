import geoopt
import torch
import numpy as np
import pytest


@pytest.mark.parametrize(
    'params',
    [
        dict(lr=1e-2),
        dict(lr=1e-3, momentum=.9),
        dict(momentum=.9, nesterov=True, lr=1e-3),
        dict(momentum=.9, dampening=.1, lr=1e-3)
    ]
)
def test_rsgd_stiefel(params):
    stiefel = geoopt.manifolds.Stiefel()
    torch.manual_seed(42)
    X = geoopt.ManifoldParameter(torch.randn(20, 10), manifold=stiefel).proj_()
    Xstar = torch.randn(20, 10)
    Xstar.set_(stiefel.projx(Xstar))

    def closure():
        optim.zero_grad()
        loss = (X - Xstar).pow(2).sum()
        # manifold constraint that makes optimization hard if violated
        loss += (X.t() @ X - torch.eye(X.shape[1])).pow(2).sum() * 100
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianSGD([X], **params)
    assert (X - Xstar).norm() > 1e-5
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-5:
            break
        optim.step(closure)

    np.testing.assert_allclose(X.data, Xstar, atol=1e-5)


def test_init_manifold():
    torch.manual_seed(42)
    stiefel = geoopt.manifolds.Stiefel()
    rn = geoopt.manifolds.Rn()
    x0 = torch.randn(10, 10)
    x1 = torch.randn(10, 10)
    p0 = geoopt.ManifoldParameter(x0, manifold=stiefel)
    p1 = geoopt.ManifoldParameter(x1, manifold=rn)
    p0.grad = torch.zeros_like(p0)
    p1.grad = torch.zeros_like(p1)
    p0old = p0.clone()
    p1old = p1.clone()
    opt = geoopt.optim.RiemannianSGD([p0, p1], lr=1, stabilize=1)
    opt.zero_grad()
    opt.step()
    assert not np.allclose(p0.data, p0old.data)
    np.testing.assert_allclose(p1.data, p1old.data)
    np.testing.assert_allclose(p0.data, stiefel.projx(p0old.data))
