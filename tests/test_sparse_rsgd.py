import geoopt
import torch
import numpy as np
import pytest


@pytest.mark.parametrize(
    "params",
    [
        dict(lr=1e-2),
        dict(lr=1e-3, momentum=0.9),
        dict(momentum=0.9, nesterov=True, lr=1e-3),
        dict(momentum=0.9, dampening=0.1, lr=1e-3),
    ],
)
def test_adam_poincare(params):
    torch.manual_seed(44)
    manifold = geoopt.PoincareBall()
    ideal = manifold.random(10, 2)
    start = manifold.random(10, 2)
    start = geoopt.ManifoldParameter(start, manifold=manifold)

    def closure():
        idx = torch.randint(10, size=(3,))
        start_select = torch.nn.functional.embedding(idx, start, sparse=True)
        ideal_select = torch.nn.functional.embedding(idx, ideal, sparse=True)
        optim.zero_grad()
        loss = manifold.dist2(start_select, ideal_select).sum()
        loss.backward()
        assert start.grad.is_sparse
        return loss.item()

    optim = geoopt.optim.SparseRiemannianSGD([start], **params)

    for _ in range(2000):
        optim.step(closure)
    np.testing.assert_allclose(start.data, ideal, atol=1e-5, rtol=1e-5)


def test_incorrect_init():
    manifold = geoopt.PoincareBall()
    param = manifold.random(2, 10, 2).requires_grad_()
    with pytest.raises(ValueError) as e:
        geoopt.optim.SparseRiemannianSGD([param], lr=1)
    assert e.match("should be matrix valued")
