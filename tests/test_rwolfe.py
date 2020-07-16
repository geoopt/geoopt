import sys
import geoopt
import torch
import numpy as np
import pytest


@pytest.mark.parametrize(
    "params", [dict(), dict(c1=1e-3, c2=0.1), dict(line_search_method="armijo")],
)
def test_rwolfe_stiefel(params):
    # Use line search to solve orthogonal procrustes
    stiefel = geoopt.manifolds.Stiefel()
    torch.manual_seed(42)
    (n, m) = (10, 20)

    A = torch.randn(n, m, dtype=torch.float64)
    Q = stiefel.random((n, n), dtype=torch.float64)
    B = Q @ A

    with torch.no_grad():
        X = stiefel.random((n, n), dtype=torch.float64)
        X.requires_grad = True

    def closure():
        optim.zero_grad()
        loss = (X @ A - B).norm()
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianLineSearch([X], **params)
    losses = []
    for i in range(1000):
        losses.append(optim.step(closure))
        if losses[-1] < 1e-4:
            break
    assert closure() < 1e-4

    optim.step(closure)
