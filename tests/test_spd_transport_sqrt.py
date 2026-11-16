import geoopt
import pytest
import torch
from scipy.linalg import sqrtm


def make_spd(shape, dtype):
    raw = torch.randn(*shape, dtype=dtype)
    return raw @ raw.transpose(-1, -2) + torch.eye(shape[-1], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_transport_matches_principal_congruence(dtype, batch):
    torch.manual_seed(1729)
    manifold = geoopt.SymmetricPositiveDefinite()
    x = make_spd((*batch, 3, 3), dtype)
    y = make_spd((*batch, 3, 3), dtype)
    raw = torch.randn(*batch, 3, 3, dtype=dtype)
    v = (raw + raw.transpose(-1, -2)) / 2
    actual = manifold.transp(x, y, v)
    expected = []
    for xi, yi, vi in zip(
        x.reshape(-1, 3, 3), y.reshape(-1, 3, 3), v.reshape(-1, 3, 3)
    ):
        congruence = torch.from_numpy(
            sqrtm(torch.linalg.solve(xi.double(), yi.double(), left=False).numpy()).real
        )
        expected.append(congruence @ vi.double() @ congruence.T)
    expected = torch.stack(expected).reshape(actual.shape).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_transport_preserves_metric_and_round_trip(dtype, batch):
    torch.manual_seed(1730)
    manifold = geoopt.SymmetricPositiveDefinite()
    x = make_spd((*batch, 3, 3), dtype)
    y = make_spd((*batch, 3, 3), dtype)
    raw = torch.randn(*batch, 3, 3, dtype=dtype)
    v = (raw + raw.transpose(-1, -2)) / 2
    transported = manifold.transp(x, y, v)
    torch.testing.assert_close(
        transported, transported.transpose(-1, -2), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        manifold.inner(x, v), manifold.inner(y, transported), rtol=4e-5, atol=1e-5
    )
    torch.testing.assert_close(
        manifold.transp(y, x, transported), v, rtol=4e-5, atol=2e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transport_diagonal_closed_form_and_broadcast(dtype):
    manifold = geoopt.SymmetricPositiveDefinite()
    x_diag = torch.tensor([1.0, 4.0, 9.0], dtype=dtype)
    y_diag = torch.tensor([[4.0, 9.0, 16.0], [16.0, 25.0, 36.0]], dtype=dtype)
    v = torch.tensor([[2.0, 1.0, -0.5], [1.0, 3.0, 0.2], [-0.5, 0.2, 4.0]], dtype=dtype)
    scaling = (y_diag / x_diag).sqrt()
    expected = scaling.unsqueeze(-1) * v * scaling.unsqueeze(-2)
    actual = manifold.transp(torch.diag(x_diag), torch.diag_embed(y_diag), v)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transport_zero_vector_and_identity(dtype):
    manifold = geoopt.SymmetricPositiveDefinite()
    x = torch.eye(3, dtype=dtype)
    y = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=dtype))
    zero = torch.zeros_like(x)
    torch.testing.assert_close(manifold.transp(x, y, zero), zero)
    torch.testing.assert_close(manifold.transp(x, x, y), y)


def test_transport_gradients_for_both_points_and_vector():
    torch.manual_seed(1731)
    manifold = geoopt.SymmetricPositiveDefinite()
    x_raw = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    y_raw = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    v_raw = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

    def transport(a, b, c):
        eye = torch.eye(3, dtype=torch.float64)
        x = a @ a.T + eye
        y = b @ b.T + eye
        v = (c + c.T) / 2
        return manifold.transp(x, y, v)

    assert torch.autograd.gradcheck(transport, (x_raw, y_raw, v_raw))
