import pytest
import torch

from geoopt.samplers import RHMC


@pytest.mark.parametrize("offset,expected", [(1000.0, 1.0), (float("nan"), 0.0)])
def test_extreme_proposals(offset, expected):
    p = torch.nn.Parameter(torch.zeros(1))
    sampler = RHMC([p])
    sampler.burnin = False
    offsets = iter([0.0, offset])

    sampler.step(lambda: p.sum() * 0 + next(offsets))

    assert sampler.acceptance_probs == [expected]
    if not expected:
        assert p.item() == 0.0
        assert sampler.log_probs == [0.0]
        assert sampler.n_rejected == 1


def test_step_uses_only_its_closure_gradient():
    def run(stale):
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.zeros(1))
        sampler = RHMC([p])
        p.grad = torch.full_like(p, stale)
        with torch.no_grad():
            sampler.step(lambda: p.sum() * 0)
        assert sampler.log_probs == []
        return p.detach()

    torch.testing.assert_close(run(0.0), run(100.0))
