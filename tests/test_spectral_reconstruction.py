import pytest
import torch

from geoopt import linalg


def _positive_matrix(batch_shape, dtype):
    generator = torch.Generator().manual_seed(42)
    data = torch.randn(*batch_shape, 4, 4, dtype=dtype, generator=generator)
    return data @ data.transpose(-1, -2) + torch.diag(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("operation", ["log", "sqrt", "inverse_sqrt"])
def test_spectral_matrix_functions_satisfy_defining_identities(dtype, batch_shape, operation):
    matrix = _positive_matrix(batch_shape, dtype)
    identity = torch.eye(4, dtype=dtype).expand_as(matrix)

    if operation == "log":
        actual = torch.matrix_exp(linalg.sym_logm(matrix))
        expected = matrix
    elif operation == "sqrt":
        root = linalg.sym_sqrtm(matrix)
        actual = root @ root
        expected = matrix
    else:
        root = linalg.sym_inv_sqrtm1(matrix)
        actual = root @ matrix @ root
        expected = identity

    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_paired_roots_and_trace_gradients(batch_shape):
    matrix = _positive_matrix(batch_shape, torch.float64).requires_grad_()
    inverse_root, root = linalg.sym_inv_sqrtm2(matrix)
    identity = torch.eye(4, dtype=matrix.dtype).expand_as(matrix)

    torch.testing.assert_close(root @ root, matrix)
    torch.testing.assert_close(inverse_root @ root, identity)
    gradient = torch.autograd.grad(linalg.trace(root).sum(), matrix, retain_graph=True)[0]
    torch.testing.assert_close(gradient, 0.5 * inverse_root)
    gradient = torch.autograd.grad(linalg.trace(inverse_root).sum(), matrix)[0]
    torch.testing.assert_close(gradient, -0.5 * torch.linalg.inv(matrix) @ inverse_root)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_custom_matrix_square_preserves_values_and_gradients(batch_shape):
    matrix = _positive_matrix(batch_shape, torch.float64).requires_grad_()
    actual = linalg.sym_funcm(matrix, torch.square)
    expected = matrix @ matrix

    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.square().sum(), matrix)[0]
    torch.testing.assert_close(gradient, 4 * matrix @ matrix @ matrix)


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_spectral_functions_on_repeated_eigenvalues(batch_shape):
    matrix = 4 * torch.eye(4, dtype=torch.float64).expand(*batch_shape, 4, 4)
    expected_root = 2 * torch.eye(4, dtype=matrix.dtype).expand_as(matrix)
    inverse_root, root = linalg.sym_inv_sqrtm2(matrix)

    torch.testing.assert_close(root, expected_root)
    torch.testing.assert_close(inverse_root, expected_root / 4)
    torch.testing.assert_close(linalg.sym_sqrtm(matrix), expected_root)


def test_spectral_functions_accept_transposed_and_empty_batches():
    matrix = _positive_matrix((2, 3), torch.float64).transpose(0, 1)
    assert not matrix.is_contiguous()
    root = linalg.sym_sqrtm(matrix)
    torch.testing.assert_close(root @ root, matrix)

    empty = torch.empty(0, 4, 4, dtype=torch.float64)
    assert linalg.sym_logm(empty).shape == empty.shape
    assert linalg.sym_inv_sqrtm1(empty).shape == empty.shape
    inverse_root, root = linalg.sym_inv_sqrtm2(empty)
    assert root.shape == empty.shape
    assert inverse_root.shape == empty.shape
