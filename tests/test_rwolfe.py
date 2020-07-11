import sys
import geoopt
import torch
import numpy as np
import pytest


@pytest.mark.parametrize(
    "params", [dict(), dict(c1=1e-3, c2=0.1)],
)
def test_rwolfe_stiefel(params):
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

    optim = geoopt.optim.RiemannianLineSearch([X], stabilize=4500, **params)
    assert (X - Xstar).norm() > 1e-5
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-5:
            break
        optim.step(closure)
    assert X.is_contiguous()
    np.testing.assert_allclose(X.data, Xstar, atol=1e-5, rtol=1e-5)
    optim.step(closure)
