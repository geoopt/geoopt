import geoopt
import torch
import numpy as np
import pytest


@pytest.mark.parametrize("params", [dict(lr=1e-2), dict(lr=1, amsgrad=True)])
def test_adam_lorentz(params):
    lorentz = geoopt.manifolds.Lorentz(k=torch.Tensor([1.0]))
    torch.manual_seed(42)
    with torch.no_grad():
        X = geoopt.ManifoldParameter(torch.randn(20, 10), manifold=lorentz).proj_()
    Xstar = torch.randn(20, 10)
    Xstar.set_(lorentz.projx(Xstar))

    def closure():
        optim.zero_grad()
        loss = (Xstar - X).pow(2).sum()
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianAdam([X], stabilize=4500, **params)
    assert optim.param_groups[0]["stabilize"] == 4500
    for _ in range(10000):
        if (Xstar - X).norm() < 1e-5:
            break
        optim.step(closure)
    assert X.is_contiguous()
    np.testing.assert_allclose(X.data, Xstar, atol=1e-5, rtol=1e-5)
    optim.load_state_dict(optim.state_dict())
    optim.step(closure)


@pytest.mark.parametrize("params", [dict(lr=1e-2), dict(lr=1, amsgrad=True)])
def test_adam_stiefel(params):
    stiefel = geoopt.manifolds.Stiefel()
    torch.manual_seed(42)
    with torch.no_grad():
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

    optim = geoopt.optim.RiemannianAdam([dict(params=[X], stabilize=4500)], **params)
    assert optim.param_groups[0]["stabilize"] == 4500
    assert (X - Xstar).norm() > 1e-5
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-5:
            break
        optim.step(closure)
    assert X.is_contiguous()
    np.testing.assert_allclose(X.data, Xstar, atol=1e-5, rtol=1e-5)
    optim.load_state_dict(optim.state_dict())
    optim.step(closure)


def test_adam_poincare():
    torch.manual_seed(44)
    manifold = geoopt.PoincareBall()
    ideal = torch.tensor([0.5, 0.5])
    start = torch.randn(2) / 2
    start = manifold.expmap0(start)
    start = geoopt.ManifoldParameter(start, manifold=manifold)

    def closure():
        optim.zero_grad()
        loss = manifold.dist(start, ideal) ** 2
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianAdam([start], lr=1e-2)

    for _ in range(2000):
        optim.step(closure)
    np.testing.assert_allclose(start.data, ideal, atol=1e-5, rtol=1e-5)
