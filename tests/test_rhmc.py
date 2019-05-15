import geoopt
import torch
import numpy as np
import pytest

import geoopt.samplers.rhmc


@pytest.fixture(autouse=True)
def withdtype():
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize(
    "params",
    [
        dict(epsilon=1e-3, n_steps=1),
        dict(epsilon=1e-4, n_steps=1),
        dict(epsilon=1e-3, n_steps=2),
        dict(epsilon=1e-4, n_steps=2),
        dict(epsilon=1e-3, n_steps=3),
        dict(epsilon=1e-4, n_steps=3),
        dict(epsilon=1e-4, n_steps=10),
        dict(epsilon=1e-5, n_steps=10),
    ],
)
def test_leapfrog_reversibility(params):
    class NormalDist(torch.nn.Module):
        def __init__(self, mu, sigma):
            super().__init__()
            self.d = torch.distributions.Normal(mu, sigma)
            self.x = torch.nn.Parameter(torch.randn_like(mu))

        def forward(self):
            return self.d.log_prob(self.x).sum()

    epsilon, n_steps = params["epsilon"], params["n_steps"]

    torch.manual_seed(42)
    nd = NormalDist(torch.randn([10]), torch.ones([10]))

    init_x = nd.x.data.numpy().copy()

    torch.manual_seed(42)
    sampler = geoopt.samplers.rhmc.RHMC(nd.parameters(), **params)

    r = torch.randn([10])

    for i in range(n_steps):
        logp = nd()
        logp.backward()
        with torch.no_grad():
            sampler._step(nd.x, r, epsilon)
            nd.x.grad.zero_()

    for i in range(n_steps):
        logp = nd()
        logp.backward()
        with torch.no_grad():
            sampler._step(nd.x, r, -epsilon)
            nd.x.grad.zero_()

    new_x = nd.x.data.numpy().copy()
    np.testing.assert_allclose(init_x, new_x, rtol=1e-5)


@pytest.mark.parametrize(
    "params",
    [
        dict(sampler="RHMC", epsilon=0.2, n_steps=5, n_burn=1000, n_samples=5000),
        dict(sampler="RSGLD", epsilon=1e-3, n_burn=3000, n_samples=10000),
        dict(
            sampler="SGRHMC",
            epsilon=1e-3,
            n_steps=1,
            alpha=0.5,
            n_burn=3000,
            n_samples=10000,
        ),
    ],
)
def test_sampling(params):
    class NormalDist(torch.nn.Module):
        def __init__(self, mu, sigma):
            super().__init__()
            self.d = torch.distributions.Normal(mu, sigma)
            self.x = torch.nn.Parameter(torch.randn_like(mu))

        def forward(self):
            return self.d.log_prob(self.x).sum()

    torch.manual_seed(42)
    D = 2
    n_burn, n_samples = params.pop("n_burn"), params.pop("n_samples")

    mu = torch.randn([D])
    sigma = torch.randn([D]).abs()

    nd = NormalDist(mu, sigma)
    Sampler = getattr(geoopt.samplers, params.pop("sampler"))
    sampler = Sampler(nd.parameters(), **params)

    for _ in range(n_burn):
        sampler.step(nd)

    points = []
    sampler.burnin = False

    for _ in range(n_samples):
        sampler.step(nd)
        points.append(nd.x.detach().numpy().copy())

    points = np.asarray(points)
    points = points[::20]
    assert nd.x.is_contiguous()
    np.testing.assert_allclose(mu.numpy(), points.mean(axis=0), atol=1e-1)
    np.testing.assert_allclose(sigma.numpy(), points.std(axis=0), atol=1e-1)


@pytest.mark.parametrize(
    "params",
    [
        dict(sampler="RHMC", epsilon=0.2, n_steps=5, n_burn=10, n_samples=50, nd=()),
        dict(sampler="RSGLD", epsilon=1e-3, n_burn=30, n_samples=100, nd=()),
        dict(
            sampler="SGRHMC",
            epsilon=1e-3,
            n_steps=1,
            alpha=0.5,
            n_burn=30,
            n_samples=100,
            nd=(3,),
        ),
        dict(
            sampler="RHMC", epsilon=0.2, n_steps=5, n_burn=1000, n_samples=50, nd=(3,)
        ),
        dict(sampler="RSGLD", epsilon=1e-3, n_burn=30, n_samples=10000, nd=(3,)),
        dict(
            sampler="SGRHMC",
            epsilon=1e-3,
            n_steps=1,
            alpha=0.5,
            n_burn=30,
            n_samples=100,
            nd=(3,),
        ),
    ],
)
def test_sampling_manifold(params):
    # should just work (all business stuff is checked above)
    class NormalDist(torch.nn.Module):
        def __init__(self, mu, sigma):
            super().__init__()
            self.d = torch.distributions.Normal(mu, sigma)
            self.x = geoopt.ManifoldParameter(
                torch.randn_like(mu), manifold=geoopt.Stiefel()
            )

        def forward(self):
            return self.d.log_prob(self.x).sum()

    torch.manual_seed(42)
    D = (5, 4)
    n_burn, n_samples = params.pop("n_burn"), params.pop("n_samples")
    nd = params.pop("nd")  # type: tuple
    mu = torch.randn(nd + D)
    sigma = torch.randn(nd + D).abs()

    nd = NormalDist(mu, sigma)
    Sampler = getattr(geoopt.samplers, params.pop("sampler"))
    sampler = Sampler(nd.parameters(), **params)

    for _ in range(n_burn):
        sampler.step(nd)

    points = []
    sampler.burnin = False

    for _ in range(n_samples):
        sampler.step(nd)
        points.append(nd.x.detach().numpy().copy())
