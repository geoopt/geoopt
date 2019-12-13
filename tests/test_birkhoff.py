import geoopt
import torch
import numpy as np
import pytest


"""
    This file puts the Birkhoff Polytope, the manifold of doubly stochastic matrices, to test.
"""


@pytest.mark.parametrize("params", [dict(lr=1e-2)])
def test_adam_birkhoff(params):
    birkhoff = geoopt.manifolds.BirkhoffPolytope()
    torch.manual_seed(42)
    with torch.no_grad():
        X = geoopt.ManifoldParameter(torch.rand(1, 5, 5), manifold=birkhoff).proj_()
    Xstar = torch.rand(1, 5, 5)
    Xstar.set_(birkhoff.projx(Xstar))

    def closure():
        optim.zero_grad()
        loss = (X - Xstar).pow(2).sum()
        # manifold constraint that makes optimization hard if violated
        row_penalty = ((X.transpose(1, 2) @ X).sum(dim=1) - 1.0).pow(2).sum() * 100
        col_penalty = ((X.transpose(1, 2) @ X).sum(dim=2) - 1.0).pow(2).sum() * 100
        loss += row_penalty + col_penalty
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianAdam([X], stabilize=4500, **params)

    assert (X - Xstar).norm() > 1e-3
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-3:
            break
        optim.step(closure)
    assert X.is_contiguous()

    np.testing.assert_allclose(X.data, Xstar, atol=1e-3, rtol=1e-3)
