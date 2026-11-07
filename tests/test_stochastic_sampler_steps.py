import pytest
import torch

from geoopt.samplers import RSGLD, SGRHMC


@pytest.mark.parametrize("sampler_type", [RSGLD, SGRHMC])
def test_sampler_closure_under_no_grad(sampler_type):
    point = torch.nn.Parameter(torch.tensor([1.0]))
    sampler = sampler_type([point])
    sampler.burnin = False
    calls = []

    def closure():
        calls.append(point.item())
        return -point.square().sum()

    with torch.no_grad():
        sampler.step(closure)

    assert calls == [1.0]
    assert sampler.log_probs == [-1.0]


def test_sgrhmc_moves_with_current_force(monkeypatch):
    point = torch.nn.Parameter(torch.zeros(1))
    sampler = SGRHMC([point], epsilon=0.25, alpha=1.0, n_steps=2)
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    positions = []

    def closure():
        positions.append(point.item())
        return point.sum()

    sampler.step(closure)
    assert positions == [0.0, 0.25]
    assert point.item() == 0.5


def test_sgrhmc_rejects_empty_trajectory():
    with pytest.raises(ValueError, match="n_steps"):
        SGRHMC([torch.nn.Parameter(torch.zeros(1))], n_steps=0)
