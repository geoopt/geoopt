"""Independent Stein-divergence values, invariances and differential oracles."""

import math
from fractions import Fraction

import geoopt
import pytest
import torch


DTYPES = [torch.float32, torch.float64]


@pytest.fixture
def stein():
    with pytest.warns(UserWarning, match="not fully implemented"):
        return geoopt.SymmetricPositiveDefinite(default_metric="SM")


def diagonal_oracle(first, second):
    # Ratio form avoids a determinant, a matrix product, and an overflowing sum.
    result = 0.0
    for a, b in zip(first, second):
        ratio_root = math.sqrt(a / b)
        result += math.log(0.5 * (ratio_root + 1.0 / ratio_root))
    return result


@pytest.mark.parametrize("dtype", DTYPES)
def test_common_covariance_scale_preserves_diagonal_value(stein, dtype):
    x_values, y_values = [1.0, 2.0, 3.0], [4.0, 1.0, 2.0]
    expected = torch.tensor(diagonal_oracle(x_values, y_values), dtype=dtype)
    scales = [1e-20, 1.0, 1e20] if dtype == torch.float32 else [1e-100, 1.0, 1e100]
    rtol, atol = (2e-5, 2e-5) if dtype == torch.float32 else (2e-12, 2e-12)
    for scale in scales:
        x = torch.diag(torch.tensor([scale * value for value in x_values], dtype=dtype))
        y = torch.diag(torch.tensor([scale * value for value in y_values], dtype=dtype))
        actual = stein.dist(x, y)
        assert torch.isfinite(actual)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(stein.dist(y, x), actual, rtol=0, atol=0)
        torch.testing.assert_close(stein.dist(x, x), torch.zeros((), dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_arithmetic_mean_stays_finite_near_dtype_maximum(stein, dtype):
    scale = 1e38 if dtype == torch.float32 else 5e307
    first, second = [2.4 * scale, 2.0 * scale], [2.6 * scale, 2.5 * scale]
    x, y = [torch.diag(torch.tensor(values, dtype=dtype)) for values in (first, second)]
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    expected = torch.tensor(diagonal_oracle(first, second), dtype=dtype)
    actual = stein.dist(x, y)
    assert torch.isfinite(actual)
    torch.testing.assert_close(
        actual, expected, rtol=2e-5, atol=2e-5 if dtype == torch.float32 else 2e-12
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_noncommuting_covariances_obey_congruence_invariance(stein, dtype):
    x = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=dtype)
    y = torch.tensor([[1.0, 0.125], [0.125, 3.0]], dtype=dtype)
    transform = torch.tensor([[1.2, 0.3], [-0.4, 0.8]], dtype=dtype)
    # Exact scalar 2x2 determinants provide an independent base value.
    det_x = Fraction(2) - Fraction(1, 4) ** 2
    det_y = Fraction(3) - Fraction(1, 8) ** 2
    det_mean = Fraction(3, 2) * 2 - Fraction(3, 16) ** 2
    expected = torch.tensor(
        math.log(float(det_mean)) - 0.5 * math.log(float(det_x * det_y)), dtype=dtype
    )
    actual = stein.dist(x, y)
    transformed = stein.dist(transform @ x @ transform.T, transform @ y @ transform.T)
    tolerance = 1e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(transformed, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
def test_broadcast_noncontiguous_covariances_and_keepdim(stein, dtype):
    first_values = [[1e-20, 2e-20, 3e-20], [1e20, 2e20, 3e20]]
    second_values = [[4e-20, 1e-20, 2e-20], [4.0, 1.0, 2.0], [4e20, 1e20, 2e20]]
    first = torch.diag_embed(torch.tensor(first_values, dtype=dtype))[:, None]
    second = torch.diag_embed(torch.tensor(second_values, dtype=dtype))[None]
    first_storage = torch.zeros(2, 1, 6, 6, dtype=dtype)
    second_storage = torch.zeros(1, 3, 6, 6, dtype=dtype)
    x, y = first_storage[..., ::2, ::2], second_storage[..., ::2, ::2]
    x.copy_(first)
    y.copy_(second)
    assert not x.is_contiguous() and not y.is_contiguous()
    expected = torch.tensor(
        [[diagonal_oracle(a, b) for b in second_values] for a in first_values], dtype=dtype
    )
    actual = stein.dist(x, y)
    kept = stein.dist(x, y, keepdim=True)
    assert actual.shape == (2, 3) and kept.shape == (2, 3, 1, 1)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(kept[..., 0, 0], actual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_diagonal_gradients_match_closed_form(stein, dtype):
    first, second = [1.0, 2.0, 3.0], [4.0, 1.0, 2.0]
    expected_x = torch.diag(
        torch.tensor([1 / (a + b) - 0.5 / a for a, b in zip(first, second)], dtype=dtype)
    )
    expected_y = torch.diag(
        torch.tensor([1 / (a + b) - 0.5 / b for a, b in zip(first, second)], dtype=dtype)
    )
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    for scale in [1e-20, 1.0, 1e20]:
        x = torch.diag(
            torch.tensor([scale * value for value in first], dtype=dtype)
        ).requires_grad_()
        y = torch.diag(
            torch.tensor([scale * value for value in second], dtype=dtype)
        ).requires_grad_()
        dx, dy = torch.autograd.grad(stein.dist(x, y), (x, y))
        assert torch.isfinite(dx).all() and torch.isfinite(dy).all()
        torch.testing.assert_close(dx * scale, expected_x, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(dy * scale, expected_y, rtol=tolerance, atol=tolerance)


def test_symmetric_directional_jacobian_and_hessian(stein):
    x = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.float64)
    y = torch.tensor([[1.0, 0.125], [0.125, 3.0]], dtype=torch.float64)
    h = torch.tensor([[0.3, -0.1], [-0.1, 0.2]], dtype=torch.float64)
    k = torch.tensor([[-0.2, 0.05], [0.05, 0.4]], dtype=torch.float64)
    coordinates = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def objective(theta):
        return stein.dist(x + theta[0] * h, y + theta[1] * k)

    inverse_sum, inverse_x, inverse_y = [torch.linalg.inv(value) for value in (x + y, x, y)]

    def trace(value):
        return torch.diagonal(value).sum()

    expected_gradient = torch.stack(
        (trace((inverse_sum - 0.5 * inverse_x) @ h), trace((inverse_sum - 0.5 * inverse_y) @ k))
    )
    hh = -trace(inverse_sum @ h @ inverse_sum @ h) + 0.5 * trace(inverse_x @ h @ inverse_x @ h)
    kk = -trace(inverse_sum @ k @ inverse_sum @ k) + 0.5 * trace(inverse_y @ k @ inverse_y @ k)
    hk = -trace(inverse_sum @ h @ inverse_sum @ k)
    expected_hessian = torch.stack((torch.stack((hh, hk)), torch.stack((hk, kk))))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(objective, coordinates),
        expected_gradient,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(objective, coordinates),
        expected_hessian,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.autograd.gradcheck(objective, (coordinates,))
    assert torch.autograd.gradgradcheck(objective, (coordinates,))


@pytest.mark.parametrize("dtype", DTYPES)
def test_well_scaled_value_retains_existing_metric_definition(stein, dtype):
    first, second = [0.5, 2.0, 5.0], [1.5, 3.0, 2.0]
    x, y = [torch.diag(torch.tensor(values, dtype=dtype)) for values in (first, second)]
    expected = torch.tensor(diagonal_oracle(first, second), dtype=dtype)
    torch.testing.assert_close(
        stein.dist(x, y),
        expected,
        rtol=1e-5 if dtype == torch.float32 else 2e-12,
        atol=1e-6 if dtype == torch.float32 else 2e-12,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_smallest_subnormal_identity_keeps_representable_mean(stein, dtype):
    # This well-conditioned 1x1 SPD value requires no unrepresentable inverse
    # or derivative. Half-first would round both summands to zero independently.
    zero = torch.zeros((), dtype=dtype)
    tiny = torch.nextafter(zero, torch.ones((), dtype=dtype))
    assert tiny.item() > 0.0
    x = tiny.reshape(1, 1)
    actual = stein.dist(x, x)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, zero, rtol=0.0, atol=0.0)
    unit = torch.ones(1, 1, dtype=dtype)
    torch.testing.assert_close(stein.dist(unit, unit), zero, rtol=0.0, atol=0.0)
