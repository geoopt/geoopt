import geoopt
import torch
import numpy as np
import pytest


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
    torch.set_default_dtype(torch.float64)

    class NormalDist(torch.nn.Module):
        def __init__(self, mu, sigma):
            super().__init__()
            self.d = torch.distributions.Normal(mu, sigma)
            self.x = torch.nn.Parameter(torch.randn_like(mu))

        
        def forward(self):
            return self.d.log_prob(self.x).sum()

    epsilon, n_steps = params['epsilon'], params['n_steps']

    torch.manual_seed(42)
    nd = NormalDist(torch.randn([10]), torch.ones([10]))

    init_x = nd.x.data.numpy().copy()

    torch.manual_seed(42)
    sampler = geoopt.samplers.RHMC(nd.parameters(), **params)

    r = torch.randn([10])

    for i in range(n_steps):
        logp = nd()
        logp.backward()
        sampler._step(nd.x, r, epsilon)
        nd.x.grad.zero_()
    
    for i in range(n_steps):
        logp = nd()
        logp.backward()
        sampler._step(nd.x, r, -epsilon)
        nd.x.grad.zero_()

    new_x = nd.x.data.numpy().copy()
    np.testing.assert_allclose(init_x, new_x, rtol=1e-5)
