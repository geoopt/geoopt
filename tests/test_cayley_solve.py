import pytest
import torch

from geoopt import BoundedDomain, UpperHalf
from geoopt.manifolds.siegel import csym_math as sm


def sample_points(dtype, batch=()):
    generator = torch.Generator().manual_seed(8927)
    raw = torch.randn(*batch, 4, 4, dtype=dtype, generator=generator)
    symmetric = (raw + raw.transpose(-1, -2)) * 0.5
    bounded = symmetric / (2 * torch.linalg.matrix_norm(symmetric, keepdim=True))
    y = symmetric.imag @ symmetric.imag.transpose(-1, -2)
    y = y + torch.eye(4, dtype=y.dtype)
    upper = torch.complex(symmetric.real, y)
    return bounded, upper


def inverse_formula(z, inverse):
    eye = torch.eye(z.shape[-1], dtype=z.dtype, device=z.device)
    if inverse:
        return (z - 1j * eye) @ torch.linalg.inv(z + 1j * eye)
    return 1j * (eye + z) @ torch.linalg.inv(eye - z)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch", [(), (3,), (2, 1)])
def test_cayley_solve_values_and_round_trip(inverse, dtype, batch):
    bounded, upper = sample_points(dtype, batch)
    point = upper if inverse else bounded
    transform = sm.inverse_cayley_transform if inverse else sm.cayley_transform
    undo = sm.cayley_transform if inverse else sm.inverse_cayley_transform
    expected = inverse_formula(point, inverse)
    actual = transform(point)
    tol = 2e-5 if dtype == torch.complex64 else 1e-10
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(undo(actual), point, atol=tol, rtol=tol)
    target = BoundedDomain() if inverse else UpperHalf()
    assert target.check_point_on_manifold(actual, atol=tol, rtol=tol)


@pytest.mark.parametrize("inverse", [False, True])
def test_cayley_solve_gradient_matches_inverse_equations(inverse):
    bounded, upper = sample_points(torch.complex128, (2,))
    point = (upper if inverse else bounded).requires_grad_()
    weight = torch.arange(point.numel(), dtype=torch.float64).reshape(point.shape)
    weight = torch.complex(weight, weight.flip(-1)) / point.numel()
    transform = sm.inverse_cayley_transform if inverse else sm.cayley_transform
    actual = transform(point)
    expected = inverse_formula(point, inverse)
    actual_grad = torch.autograd.grad((actual * weight).real.sum(), point)[0]
    expected_grad = torch.autograd.grad((expected * weight).real.sum(), point)[0]
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("inverse", [False, True])
def test_cayley_diagonal_frechet_derivative(inverse):
    diagonal = torch.tensor([0.1 + 0.2j, -0.2 + 0.1j, 0.4 - 0.1j], dtype=torch.complex128)
    if inverse:
        diagonal = diagonal.real + 1j * (diagonal.imag + 2)
    point = torch.diag(diagonal)
    direction = torch.tensor(
        [[1 + 2j, 3 - 1j, 2j], [3 - 1j, -1 + 0.5j, 4j], [2j, 4j, 2 - 3j]],
        dtype=point.dtype,
    )
    transform = sm.inverse_cayley_transform if inverse else sm.cayley_transform
    _, actual = torch.autograd.functional.jvp(transform, (point,), (direction,))
    denominator = diagonal + 1j if inverse else 1 - diagonal
    expected = 2j * direction / denominator[:, None] / denominator[None, :]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("inverse", [False, True])
def test_cayley_noncontiguous_and_empty_batches(inverse):
    bounded, upper = sample_points(torch.complex128, (4,))
    point = (upper if inverse else bounded)[::2]
    assert not point.is_contiguous()
    transform = sm.inverse_cayley_transform if inverse else sm.cayley_transform
    torch.testing.assert_close(
        transform(point), inverse_formula(point, inverse), atol=1e-10, rtol=1e-10
    )
    assert transform(point[:0]).shape == (0, 4, 4)


@pytest.mark.parametrize("inverse", [False, True])
def test_cayley_near_origin_and_boundary(inverse):
    diagonal = torch.tensor([0.0, 1e-12, 1 - 1e-6, -1 + 1e-6], dtype=torch.complex128)
    if inverse:
        diagonal = 1j * (1 + diagonal) / (1 - diagonal)
    point = torch.diag(diagonal)
    transform = sm.inverse_cayley_transform if inverse else sm.cayley_transform
    expected = (
        (diagonal - 1j) / (diagonal + 1j) if inverse else 1j * (1 + diagonal) / (1 - diagonal)
    )
    torch.testing.assert_close(transform(point), torch.diag(expected), atol=1e-10, rtol=1e-10)
