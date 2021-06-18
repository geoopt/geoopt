import torch
import random
import numpy as np
import pytest
import geoopt.manifolds.siegel.csym_math as sm
from geoopt.linalg import sym


@pytest.fixture(scope="module", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="module", params=[torch.complex128, torch.complex64])
def dtype(request):
    return request.param


@pytest.fixture(scope="module", params=range(3, 10))
def rank(request):
    return request.param


def get_random_complex_symmetric_matrices(points: int, dims: int, dtype: torch.dtype):
    """Returns 'points' random symmetric matrices of 'dims' x 'dims'"""
    m = torch.rand((points, dims, dims), dtype=dtype)
    return sym(m)


def assert_sds_equals_a(s, eigenvalues, a, atol=1e-5, rtol=1e-5):
    diagonal = torch.diag_embed(eigenvalues)
    diagonal = sm.to_complex(diagonal, torch.zeros_like(diagonal))
    np.testing.assert_allclose(a, s.conj() @ diagonal @ s.conj().transpose(-1, -2), atol=atol, rtol=rtol)


def test_takagi_factorization_idem_takagi_fact():
    real_a = torch.Tensor([[[0.222662581850819, 0.986442232151068],
                            [0.986442232151068, 0.279075835428948]]])
    imag_a = torch.Tensor([[[0.204990135642257, 1.28603437847092],
                            [1.28603437847092, 1.18098457998164]]])
    a = sm.to_complex(real_a, imag_a)

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_real_pos_imag_pos(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype)

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_real_pos_imag_neg(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype)
    a = sm.to_complex(a.real, a.imag * -1)

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_real_neg_imag_pos(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype)
    a = sm.to_complex(a.real * -1, a.imag)

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_real_neg_imag_neg(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype)
    a = sm.to_complex(a.real * -1, a.imag * -1)

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_small_values(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype) / 10

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_large_values(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype) * 10

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_very_large_values(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype) * 1000

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a, atol=1e-3, rtol=1e-3)


def test_takagi_factorization_real_identity(rank, dtype):
    a = sm.identity_like(get_random_complex_symmetric_matrices(10, rank, dtype))

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)
    np.testing.assert_allclose(a, s, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(torch.ones_like(eigenvalues), eigenvalues, atol=1e-5, rtol=1e-5)


def test_takagi_factorization_imag_identity(rank, dtype):
    a = sm.identity_like(get_random_complex_symmetric_matrices(10, rank, dtype))
    a = sm.to_complex(a.imag, a.real)

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)


def test_takagi_factorization_real_diagonal(rank, dtype):
    real_a = torch.diag_embed(torch.rand((10, rank), dtype=dtype).real * 10)
    a = torch.complex(real_a, torch.zeros_like(real_a))

    eigenvalues, s = sm.takagi_eig(a)

    assert_sds_equals_a(s, eigenvalues, a)
    # real part of eigenvectors is made of vectors with one 1 and all zeros
    real_part = torch.sum(torch.abs(s.real), dim=-1)
    np.testing.assert_allclose(torch.ones_like(real_part), real_part, atol=1e-5, rtol=1e-5)
    # imaginary part of eigenvectors is all zeros
    np.testing.assert_allclose(torch.zeros(1), torch.sum(s.imag), atol=1e-5, rtol=1e-5)


def test_takagi_factorization_eigval_equals_eig(rank, dtype):
    a = get_random_complex_symmetric_matrices(10, rank, dtype)

    eig, s = sm.takagi_eig(a)
    eigvals = sm.takagi_eigvals(a)

    assert_sds_equals_a(s, eigvals, a)
    np.testing.assert_allclose(eig, eigvals, atol=1e-5, rtol=1e-5)
