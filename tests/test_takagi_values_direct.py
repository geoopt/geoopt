import pytest
import torch
from geoopt.manifolds.siegel import csym_math as sm


def compound_reference(z):
    top = torch.cat((z.real, z.imag), dim=-1)
    bottom = torch.cat((z.imag, -z.real), dim=-1)
    return torch.linalg.eigvalsh(torch.cat((top, bottom), dim=-2))[..., z.shape[-1] :]


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
@pytest.mark.parametrize("values", [[0.2, 0.7, 1.4], [0.0, 0.5, 2.0], [0.5, 0.5, 0.5]])
@pytest.mark.parametrize("scale", [1e-7, 1.0, 1e7])
def test_takagi_values_known_unitary_congruence(dtype, batch, values, scale):
    torch.manual_seed(1800)
    raw = torch.randn(*batch, 3, 3, dtype=dtype)
    unitary, _ = torch.linalg.qr(raw)
    expected = scale * torch.tensor(values, dtype=raw.real.dtype).expand(*batch, 3)
    z = (unitary * expected.unsqueeze(-2)) @ unitary.transpose(-1, -2)
    actual = sm.takagi_eigvals(z)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6 * scale)
    assert torch.all(actual[..., 1:] >= actual[..., :-1])


@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
def test_takagi_values_symmetric_direction_gradients(batch):
    torch.manual_seed(1801)
    raw = torch.randn(*batch, 3, 3, dtype=torch.complex128, requires_grad=True)

    def actual(value):
        return sm.takagi_eigvals((value + value.transpose(-1, -2)) / 2)

    assert torch.autograd.gradcheck(actual, (raw,))
    z = (raw + raw.transpose(-1, -2)) / 2
    weights = torch.tensor([0.3, 0.7, 1.1], dtype=torch.float64)
    reference = torch.autograd.grad((compound_reference(z) * weights).sum(), raw)[0]
    observed = torch.autograd.grad((actual(raw) * weights).sum(), raw)[0]
    torch.testing.assert_close(observed, reference, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("batch", [(0,), (2, 0)])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_takagi_values_empty_batches(batch, dtype):
    z = torch.empty(*batch, 3, 3, dtype=dtype)
    actual = sm.takagi_eigvals(z)
    assert actual.shape == (*batch, 3)
    assert actual.dtype == z.real.dtype
