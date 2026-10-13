import geoopt
import torch
import numpy as np
import pytest


"""
    This file puts the Birkhoff Polytope, the manifold of doubly stochastic matrices, to test.
"""


@pytest.mark.parametrize("params", [dict(lr=1e-2)])
def test_adam_birkhoff(params):
    birkhoff = geoopt.manifolds.BirkhoffPolytope(tol=1e-5)
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

    optim = geoopt.optim.RiemannianAdam([X], stabilize=1000, **params)

    assert (X - Xstar).norm() > 1e-3
    for _ in range(10000):
        if (X - Xstar).norm() < 1e-3:
            break
        optim.step(closure)
    assert X.is_contiguous()

    np.testing.assert_allclose(X.data, Xstar, atol=1e-3, rtol=1e-3)


def test_check_point_rejects_negative_entries():
    birkhoff = geoopt.manifolds.BirkhoffPolytope()
    x_bad = torch.tensor(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
    )
    ok, _ = birkhoff._check_point_on_manifold(x_bad)
    assert not ok
    # a genuine doubly stochastic matrix still passes
    ok, _ = birkhoff._check_point_on_manifold(torch.eye(3))
    assert ok


def test_retr_at_vertex_permutation_matrix():
    birkhoff = geoopt.manifolds.BirkhoffPolytope()
    P = torch.eye(3)
    torch.manual_seed(0)
    u = torch.randn(3, 3)
    u = u - u.mean(dim=0, keepdim=True)
    u = u - u.mean(dim=1, keepdim=True)
    ok, reason = birkhoff._check_vector_on_tangent(P, u)
    assert ok, reason
    y = birkhoff.retr(P, u)
    assert torch.isfinite(y).all()
    assert (y >= 0).all()
    np.testing.assert_allclose(
        y.sum(dim=-1), torch.ones(3), atol=1e-4, rtol=0.0
    )
    np.testing.assert_allclose(
        y.sum(dim=-2), torch.ones(3), atol=1e-4, rtol=0.0
    )


def test_proj_doubly_stochastic_default_eps_accuracy():
    torch.manual_seed(1)
    x = torch.rand(4, 4)
    y = geoopt.manifolds.birkhoff_polytope.proj_doubly_stochastic(x)
    np.testing.assert_allclose(
        y.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        y.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=1e-5
    )
