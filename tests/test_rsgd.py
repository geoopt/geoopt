import geoopt
import torch


def test_rsgd_simple():
    stiefel = geoopt.manifolds.Stiefel()
    torch.manual_seed(42)
    X = geoopt.ManifoldParameter(torch.randn(20, 10)).proj_()
    Xstar = torch.randn(20, 10)
    Xstar.set_(stiefel.projx(Xstar))

    def closure():
        loss = (X - Xstar).norm()
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianSGD([X], 1e-3)
    for _ in range(10000):
        print(optim.step(closure), end=' ')

    assert torch.allclose(X, Xstar)
