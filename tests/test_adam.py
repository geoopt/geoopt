import geoopt
import torch
import numpy as np

def test_adam_hyperboloid():
    man = geoopt.Lorentz(k=torch.Tensor([1.]))
    torch.manual_seed(44)
    ideal = torch.randn(1, 2)
    ideal = man.projx(ideal)
    start = torch.randn(1, 2)
    start = man.projx(start)
    start = geoopt.ManifoldParameter(start, manifold=geoopt.Lorentz())

    def closure():
        optim.zero_grad()
        loss = man.dist(start, ideal)
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianAdam([start], lr=1e-3)

    for _ in range(3000):
        optim.step(closure)
        if man.dist(start, ideal) < 1e-4:
            break
    print(geoopt.manifolds.lorentz.math.dist(start, ideal, k=torch.Tensor([1.])))

test_adam_hyperboloid()
