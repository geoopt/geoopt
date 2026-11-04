import pytest
import torch

import geoopt


@pytest.mark.parametrize("k", [0.5, 1.0, 2.0])
def test_egrad2rgrad_does_not_modify_the_gradient(k):
    torch.manual_seed(0)
    manifold = geoopt.Lorentz(k=torch.tensor(k, dtype=torch.float64))
    x = manifold.random_normal(5, 4, std=0.5, dtype=torch.float64)
    grad = torch.randn(5, 4, dtype=torch.float64)
    grad_copy = grad.clone()

    rgrad = manifold.egrad2rgrad(x, grad)
    torch.testing.assert_close(grad, grad_copy, rtol=0, atol=0)
    # calling it again on the same tensor gives the same result
    torch.testing.assert_close(manifold.egrad2rgrad(x, grad), rgrad)

    # defining property of the Riemannian gradient: <rgrad, v>_x == <grad, v>
    v = manifold.proju(x, torch.randn(5, 4, dtype=torch.float64))
    torch.testing.assert_close(manifold.inner(x, rgrad, v), (grad_copy * v).sum(-1))
    # tangent: <rgrad, x>_L == 0
    torch.testing.assert_close(manifold.inner(x, rgrad, x), torch.zeros(5, dtype=torch.float64))


def test_egrad2rgrad_accepts_expanded_gradient():
    manifold = geoopt.Lorentz(k=torch.tensor(1.0, dtype=torch.float64))
    x = manifold.random_normal(3, 4, dtype=torch.float64)
    grad = torch.randn(4, dtype=torch.float64).expand(3, 4)
    rgrad = manifold.egrad2rgrad(x, grad)
    torch.testing.assert_close(rgrad, manifold.egrad2rgrad(x, grad.contiguous()))


def test_optimizer_step_keeps_param_grad():
    torch.manual_seed(0)
    manifold = geoopt.Lorentz(k=torch.tensor(1.0, dtype=torch.float64))
    target = manifold.random_normal(4, dtype=torch.float64)
    param = geoopt.ManifoldParameter(
        manifold.random_normal(4, dtype=torch.float64), manifold=manifold
    )
    optimizer = geoopt.optim.RiemannianSGD([param], lr=0.1)
    manifold.dist(param, target).pow(2).backward()
    grad = param.grad.clone()
    optimizer.step()
    torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)
