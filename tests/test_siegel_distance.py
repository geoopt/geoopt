import torch
import random
import numpy as np
import pytest
from itertools import product
from geoopt import UpperHalf, BoundedDomain
from geoopt.manifolds.siegel.vvd_metrics import SiegelMetricType
from geoopt.manifolds.siegel.csym_math import EPS


@pytest.fixture(scope="module", autouse=True, params=range(30, 34))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="module", params=[torch.complex128, torch.complex64])
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def eps(dtype):
    if dtype == torch.complex64:
        return EPS[torch.float32]
    return EPS[torch.float64]


@pytest.fixture(scope="module", params=[3, 4, 5, 7, 10])
def rank(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=product(
        [UpperHalf, BoundedDomain],
        [
            SiegelMetricType.RIEMANNIAN,
            SiegelMetricType.FINSLER_ONE,
            SiegelMetricType.FINSLER_INFINITY,
            SiegelMetricType.FINSLER_MINIMUM,
            SiegelMetricType.WEIGHTED_SUM,
        ],
    ),
)
def manifold(request, rank):
    manifold_class, metric = request.param
    return manifold_class(metric=metric, rank=rank)


def test_distance_is_symmetric_real_pos_imag_pos(manifold, rank, dtype, eps):
    x = manifold.random(10, rank, rank, dtype=dtype)
    y = manifold.random(10, rank, rank, dtype=dtype)

    dist_xy = manifold.dist(x, y)
    dist_yx = manifold.dist(y, x)

    np.testing.assert_allclose(dist_xy.detach(), dist_yx.detach(), atol=eps, rtol=eps)


def test_distance_is_symmetric_real_neg_imag_pos(manifold, rank, dtype, eps):
    x = manifold.random(10, rank, rank, dtype=dtype)
    x.real = x.real * -1
    y = manifold.random(10, rank, rank, dtype=dtype)

    dist_xy = manifold.dist(x, y).detach()
    dist_yx = manifold.dist(y, x).detach()

    np.testing.assert_allclose(dist_xy, dist_yx, atol=eps, rtol=eps)


def test_distance_to_same_point_is_zero(manifold, rank, dtype, eps):
    x = manifold.random(10, rank, rank, dtype=dtype)

    dist_xx = manifold.dist(x, x).detach()

    np.testing.assert_allclose(dist_xx, torch.zeros_like(dist_xx), atol=eps, rtol=eps)


def test_distance_to_same_point_in_some_cases_is_zero(manifold, rank, dtype, eps):
    x = manifold.random(10, rank, rank, dtype=dtype)
    y = manifold.random(10, rank, rank, dtype=dtype)
    x[5:] = y[5:]

    dist_xy = manifold.dist(x, y).detach()
    dist_yx = manifold.dist(y, x).detach()

    np.testing.assert_allclose(
        dist_xy[5:], torch.zeros_like(dist_xy[5:]), atol=eps, rtol=eps
    )
    np.testing.assert_allclose(
        dist_yx[5:], torch.zeros_like(dist_yx[5:]), atol=eps, rtol=eps
    )
    assert torch.all(dist_xy[5:] != 0).item()
    assert torch.all(dist_yx[5:] != 0).item()


def test_distance_is_symmetric_with_diagonal_matrices(manifold, rank, dtype, eps):
    x = manifold.random(10, rank, rank, dtype=dtype)
    y = manifold.random(10, rank, rank, dtype=dtype)
    id = torch.eye(rank).unsqueeze(0).repeat(10, 1, 1).bool()
    x = torch.complex(
        torch.where(id, x.real, torch.zeros_like(x.real)),
        torch.where(id, x.imag, torch.zeros_like(x.imag)),
    )
    y = torch.complex(
        torch.where(id, y.real, torch.zeros_like(y.real)),
        torch.where(id, y.imag, torch.zeros_like(y.imag)),
    )

    dist_xy = manifold.dist(x, y).detach()
    dist_yx = manifold.dist(y, x).detach()

    np.testing.assert_allclose(dist_xy, dist_yx, atol=eps, rtol=eps)


def test_distance_is_symmetric_only_imaginary_matrices(manifold, rank, dtype, eps):
    x = manifold.random(10, rank, rank, dtype=dtype)
    y = manifold.random(10, rank, rank, dtype=dtype)

    x.real = torch.zeros_like(x.real)
    y.real = torch.zeros_like(y.real)

    dist_xy = manifold.dist(x, y).detach()
    dist_yx = manifold.dist(y, x).detach()

    np.testing.assert_allclose(dist_xy, dist_yx, atol=eps, rtol=eps)


def test_distance_from_origin_to_origin_is_zero(manifold, rank, dtype, eps):
    x = manifold.origin(rank, rank, dtype=dtype)
    y = manifold.origin(rank, rank, dtype=dtype)

    dist_xy = manifold.dist(x, y).detach()

    np.testing.assert_allclose(dist_xy, torch.zeros_like(dist_xy), atol=eps, rtol=eps)
