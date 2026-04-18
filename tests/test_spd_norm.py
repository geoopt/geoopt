import torch
import pytest
import geoopt


class TestSPDNormBackward:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.manifold = geoopt.SymmetricPositiveDefinite()
        self.x = torch.eye(2, dtype=torch.float64)

    def test_zero_tangent_vector_returns_zero_gradient(self):
        u = torch.zeros((2, 2), requires_grad=True, dtype=torch.float64)
        norm = self.manifold.norm(self.x, u)
        norm.backward()
        assert torch.allclose(u.grad, torch.zeros(2, 2, dtype=torch.float64))

    def test_nonzero_tangent_vector_gradient(self):
        u = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True, dtype=torch.float64
        )
        norm = self.manifold.norm(self.x, u)
        norm.backward()
        expected = torch.tensor([[0.7071, 0.0], [0.0, 0.7071]], dtype=torch.float64)
        assert torch.allclose(u.grad, expected, atol=1e-4)

    def test_keepdim_true(self):
        u = torch.zeros((2, 2), requires_grad=True, dtype=torch.float64)
        norm = self.manifold.norm(self.x, u, keepdim=True)
        assert norm.shape == torch.Size([1, 1])
        norm.backward()
        assert torch.allclose(u.grad, torch.zeros(2, 2, dtype=torch.float64))

    def test_batch_mixed_zero_nonzero(self):
        x = torch.eye(2, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
        u = torch.zeros(2, 2, 2, dtype=torch.float64)
        u[0, 0, 0] = 1
        u.requires_grad_(True)
        norm = self.manifold.norm(x, u)
        norm.sum().backward()
        expected = torch.zeros(2, 2, 2, dtype=torch.float64)
        expected[0, 0, 0] = 1
        assert torch.allclose(u.grad, expected)

    def test_compare_with_torch_norm_behavior(self):
        u_spd = torch.zeros((2, 2), requires_grad=True, dtype=torch.float64)
        norm_spd = self.manifold.norm(self.x, u_spd)
        norm_spd.backward()

        u_torch = torch.zeros((2, 2), requires_grad=True, dtype=torch.float64)
        norm_torch = torch.norm(u_torch)
        norm_torch.backward()

        assert torch.allclose(u_spd.grad, u_torch.grad)
