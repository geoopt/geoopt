import numpy as np
import pytest
import torch

from geoopt.samplers import RHMC


@pytest.mark.parametrize("epsilons", [(0.1, 1.0), (1.0, 0.1), (0.2, 0.6), (0.6, 0.2), (0.2, 0.2)])
@pytest.mark.parametrize("n_steps", [1, 3, 7])
@pytest.mark.parametrize("seed", [42, 83])
def test_gaussian_parameter_group_acceptance(epsilons, n_steps, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Zero initial force isolates the final kick from the initial kick's size.
    parameters = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)) for _ in epsilons]
    groups = [dict(params=[p], epsilon=e) for p, e in zip(parameters, epsilons)]
    sampler = RHMC(groups, n_steps=n_steps)
    sampler.burnin = False
    positions = []

    def closure():
        positions.append(np.concatenate([p.detach().numpy().copy() for p in parameters]))
        return sum(-0.5 * p.square().sum() for p in parameters)

    sampler.step(closure)
    energy_difference = 0.0
    proposals = []
    for p, epsilon in zip(parameters, epsilons):
        momentum = sampler.state[p]["old_r"].numpy()
        kick = np.array([[1.0, 0.0], [-0.5 * epsilon, 1.0]])
        drift = np.array([[1.0, epsilon], [0.0, 1.0]])
        phase_map = np.linalg.matrix_power(kick @ drift @ kick, n_steps)
        expected = np.stack([np.zeros_like(momentum), momentum], axis=-1) @ phase_map.T
        proposals.append(expected[:, 0])
        energy_difference += 0.5 * (np.square(momentum).sum() - np.square(expected).sum())
    np.testing.assert_allclose(positions[-1], np.concatenate(proposals), atol=1e-13, rtol=1e-13)
    probability = np.exp(min(0.0, energy_difference))
    assert sampler.acceptance_probs[-1] == pytest.approx(probability, abs=1e-13)
