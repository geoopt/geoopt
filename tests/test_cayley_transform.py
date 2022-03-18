import torch
import random
import numpy as np
import pytest
import geoopt.manifolds.siegel.csym_math as sm
from geoopt.linalg import sym
from geoopt import UpperHalf, BoundedDomain


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


@pytest.fixture(scope="module")
def eps(dtype):
    if dtype == torch.complex64:
        return sm.EPS[torch.float32]
    return sm.EPS[torch.float64]


@pytest.fixture(scope="module")
def bounded(rank):
    return BoundedDomain(rank=rank)


@pytest.fixture(scope="module")
def upper(rank):
    return UpperHalf(rank=rank)


def test_cayley_transform(bounded, upper, rank, dtype, eps):
    x = bounded.random(10, rank, rank, dtype=dtype).detach()

    tran_x = sm.cayley_transform(x)
    result = sm.inverse_cayley_transform(tran_x)

    np.testing.assert_allclose(x, result, atol=eps, rtol=eps)
    bounded.assert_check_point_on_manifold(result)
    upper.assert_check_point_on_manifold(tran_x)


def test_cayley_transform_with_projx(bounded, upper, rank, dtype, eps):
    ex = torch.randn((10, rank, rank), dtype=dtype)
    ex = sym(ex)
    x = bounded.projx(ex).detach()

    tran_x = sm.cayley_transform(x)
    result = sm.inverse_cayley_transform(tran_x)

    np.testing.assert_allclose(x, result, atol=eps, rtol=eps)
    bounded.assert_check_point_on_manifold(result, atol=eps)
    upper.assert_check_point_on_manifold(tran_x, atol=eps)


def test_inverse_cayley_transform(bounded, upper, rank, dtype, eps):
    x = upper.random(10, rank, rank, dtype=dtype).detach()

    tran_x = sm.inverse_cayley_transform(x)
    result = sm.cayley_transform(tran_x)

    np.testing.assert_allclose(x, result, atol=eps, rtol=eps)
    upper.assert_check_point_on_manifold(result)
    bounded.assert_check_point_on_manifold(tran_x)


def test_inverse_cayley_transform_from_projx(bounded, upper, rank, dtype, eps):
    ex = torch.randn((10, rank, rank), dtype=dtype)
    ex = sym(ex)
    x = upper.projx(ex).detach()

    tran_x = sm.inverse_cayley_transform(x)
    result = sm.cayley_transform(tran_x)

    np.testing.assert_allclose(x, result, atol=eps, rtol=eps)
    upper.assert_check_point_on_manifold(result)
    bounded.assert_check_point_on_manifold(tran_x)
