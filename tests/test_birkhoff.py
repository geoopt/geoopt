import geoopt
import torch
import numpy as np
import pytest


"""
    This file puts the Birkhoff Polytope, the manifold of doubly stochastic matrices, to test.
"""

@pytest.mark.parametrize("params", [dict(lr=1e-2), ])
def test_adam_birkhoff(params):
    birkhoff = geoopt.manifolds.BirkhoffPolytope()
    torch.manual_seed(42)
    with torch.no_grad():
        X = geoopt.ManifoldParameter(torch.rand(10, 5, 5), manifold=birkhoff).proj_()
    Xstar = torch.rand(10, 5, 5)
    Xstar.set_(birkhoff.projx(Xstar))

    optim = geoopt.optim.RiemannianAdam([X], stabilize=4500, **params)
    for _ in range(10000):
        optim.zero_grad()
        loss = torch.norm(X - Xstar)
        if loss < 1e-3:
            break
        loss.backward()
        print('{} {}'.format(_, loss))
        optim.step()

    np.testing.assert_allclose(X.data, Xstar, atol=1e-3, rtol=1e-3)


# if __name__ == '__main__':
#     params = dict(lr=1e-2)
#     test_adam_birkhoff(params)
