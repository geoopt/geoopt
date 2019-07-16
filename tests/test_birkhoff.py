import geoopt
import torch
import numpy as np
import pytest


"""
    This file puts the Birkhoff Polytope, the manifold of doubly stochastic matrices, to test.
"""



@pytest.mark.parametrize("params", [dict(lr=1e-2), dict(lr=1, amsgrad=True)])
def test_adam_stiefel(params):
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

    optim = geoopt.optim.RiemannianAdam([X], stabilize=4500, **params)
    assert (X - Xstar).norm() > 1e-5
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-5:
            break
        optim.step(closure)

    np.testing.assert_allclose(X.data, Xstar, atol=1e-5, rtol=1e-5)
    optim.load_state_dict(optim.state_dict())
    optim.step(closure)


def test_adam_birkhoff(params):
    birkhoff = geoopt.manifolds.BirkhoffPolytope()
    torch.manual_seed(42)
    X = geoopt.ManifoldParameter(torch.rand(36, 5, 5), manifold=birkhoff).proj_()
    Xstar = torch.rand(36, 5, 5)
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


# def test_adam_birkhoff2(params):
#     birkhoff = geoopt.manifolds.BirkhoffPolytope()
#     torch.manual_seed(42)
#     P = geoopt.ManifoldParameter(torch.rand(1, 5, 5), manifold=birkhoff).proj_()
#     Xstar = torch.rand(36, 5, 5)
#     X = torch.rand(36, 5, 5)
#
#     Xstar.set_(birkhoff.projx(Xstar))
#     X.set_(birkhoff.projx(X))
#
#     optim = geoopt.optim.RiemannianAdam([P], stabilize=4500, **params)
#     for _ in range(10000):
#         optim.zero_grad()
#         loss = torch.norm(torch.matmul(P, X) - Xstar)
#         if loss < 1e-3:
#             break
#         loss.backward()
#         print('{} {}'.format(_, loss))
#         optim.step()
#
#
# def test_adam_birkhoff3(params):
#     birkhoff = geoopt.manifolds.BirkhoffPolytope()
#     torch.manual_seed(42)
#     P = torch.rand(36, 5, 5)
#     P.requires_grad = True
#     Xstar = torch.rand(36, 5, 5)
#     X = torch.rand(36, 5, 5)
#
#     Xstar.set_(birkhoff.projx(Xstar))
#     X.set_(birkhoff.projx(X))
#     from torch.optim import Adam
#     optim = Adam([P], lr=1e-2)
#     optim = geoopt.optim.RiemannianAdam([P], stabilize=4500, **params)
#     for _ in range(10000):
#         optim.zero_grad()
#         P2 = birkhoff.projx(P)
#         loss = torch.norm(torch.matmul(P2, X) - Xstar)
#         if loss < 1e-3:
#             break
#         loss.backward()
#         print('{} {}'.format(_, loss))
#         optim.step()


if __name__ == '__main__':
    params = dict(lr=1e-2)
    test_adam_birkhoff(params)
    # test_adam_birkhoff2(params)
    # test_adam_birkhoff3(params)
