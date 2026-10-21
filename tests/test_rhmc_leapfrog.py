import numpy as np
import pytest
import torch

from geoopt.samplers import RHMC


@pytest.mark.parametrize("epsilon", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("n_steps", [1, 3, 7])
def test_gaussian_leapfrog_proposal(epsilon, n_steps):
    torch.manual_seed(42)
    np.random.seed(42)
    p = torch.nn.Parameter(torch.tensor([0.75, -1.25], dtype=torch.float64))
    initial = p.detach().numpy().copy()
    positions = []

    def closure():
        positions.append(p.detach().numpy().copy())
        return -0.5 * p.square().sum()

    sampler = RHMC([p], epsilon=epsilon, n_steps=n_steps)
    sampler.burnin = False
    sampler.step(closure)

    # For H=(q**2+r**2)/2, each leapfrog step has this phase-space matrix.
    kick = np.array([[1.0, 0.0], [-0.5 * epsilon, 1.0]])
    drift = np.array([[1.0, epsilon], [0.0, 1.0]])
    phase_map = np.linalg.matrix_power(kick @ drift @ kick, n_steps)
    momentum = sampler.state[p]["old_r"].numpy()
    expected = np.stack([initial, momentum], axis=-1) @ phase_map.T
    np.testing.assert_allclose(positions[-1], expected[:, 0], rtol=1e-13, atol=1e-13)

    old_energy = 0.5 * (initial @ initial + momentum @ momentum)
    new_energy = 0.5 * np.square(expected).sum()
    expected_probability = np.exp(min(0.0, old_energy - new_energy))
    assert sampler.acceptance_probs[-1] == pytest.approx(expected_probability, abs=1e-13)
    assert len(positions) == n_steps + 1
