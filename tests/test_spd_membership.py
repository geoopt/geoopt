import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_spd_membership_preserves_tolerance_and_input_gradients(
    dtype, batch_shape, requires_grad
):
    manifold = geoopt.SymmetricPositiveDefinite()
    for minimum, expected in [(0.5, True), (0.0, True), (-0.0005, True), (-0.002, False)]:
        spectrum = torch.tensor([minimum, 1.0, 3.0], dtype=dtype)
        point = torch.diag(spectrum).expand(batch_shape + (3, 3)).clone()
        point.requires_grad_(requires_grad)
        original = point.detach().clone()
        valid, reason = manifold.check_point_on_manifold(point, explain=True, atol=0.001)
        assert valid is expected
        assert (reason is None) is expected
        assert torch.equal(point, original)
        assert point.requires_grad is requires_grad
        if requires_grad:
            point.square().sum().backward()
            torch.testing.assert_close(point.grad, 2 * point.detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spd_membership_checks_every_matrix_and_preserves_symmetry_rejection(dtype):
    manifold = geoopt.SymmetricPositiveDefinite()
    point = torch.eye(3, dtype=dtype).expand(2, 4, 3, 3).clone()
    point[1, 2, 0, 0] = -1
    assert not manifold.check_point_on_manifold(point)
    point[1, 2, 0, 0] = 1
    point[1, 2, 0, 1] = 0.2
    valid, reason = manifold.check_point_on_manifold(point, explain=True)
    assert not valid
    assert "transpose" in reason


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spd_membership_accepts_rotated_spectra_and_noncontiguous_inputs(dtype):
    manifold = geoopt.SymmetricPositiveDefinite()
    generator = torch.Generator().manual_seed(912)
    basis, _ = torch.linalg.qr(torch.randn(2, 3, 5, 5, dtype=dtype, generator=generator))
    spectrum = torch.tensor([0.01, 0.3, 1.0, 1.0, 5.0], dtype=dtype)
    point = ((basis * spectrum) @ basis.transpose(-1, -2)).transpose(0, 1)
    assert not point.is_contiguous()
    assert manifold.check_point_on_manifold(point)
    assert manifold.check_point_on_manifold(point.transpose(-1, -2))
