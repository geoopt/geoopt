"""
Tests ideas are taken mostly from https://github.com/dalab/hyperbolic_nn/blob/master/util.py with some changes
"""
import torch
import random
import numpy as np
import pytest
from geoopt.manifolds import poincare


@pytest.fixture("function", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


@pytest.fixture
def c(seed):
    # test broadcasted and non broadcasted versions
    if seed == 30:
        c = 0
    elif seed == 35:
        c = torch.zeros(100, 1, dtype=torch.float64)
    elif seed > 35:
        c = torch.rand(100, 1, dtype=torch.float64)
    else:
        c = random.random()
    return c + 1e-10


@pytest.fixture
def a(seed, c):
    if seed in {30, 35}:
        a = torch.randn(100, 10, dtype=torch.float64)
    elif seed > 35:
        # do not check numerically unstable regions
        # I've manually observed small differences there
        a = torch.empty(100, 10, dtype=torch.float64).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= (torch.rand_like(c) * c) ** 0.5
    else:
        a = torch.empty(100, 10, dtype=torch.float64).normal_(-1, 1)
        a /= a.norm(dim=-1, keepdim=True) * 1.3
        a *= random.uniform(0, c) ** 0.5
    return a


@pytest.fixture
def b(seed, c):
    if seed in {30, 35}:
        b = torch.randn(100, 10, dtype=torch.float64)
    elif seed > 35:
        b = torch.empty(100, 10, dtype=torch.float64).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= (torch.rand_like(c) * c) ** 0.5
    else:
        b = torch.empty(100, 10, dtype=torch.float64).normal_(-1, 1)
        b /= b.norm(dim=-1, keepdim=True) * 1.3
        b *= random.uniform(0, c) ** 0.5
    return b


def test_mobius_addition_left_cancelation(a, b, c):
    res = poincare.math.mobius_add(-a, poincare.math.mobius_add(a, b, c=c), c=c)
    np.testing.assert_allclose(res, b)


def test_mobius_addition_left_cancelation_broadcasted(a, b, c):
    res = poincare.math.mobius_add(-a, poincare.math.mobius_add(a, b, c=c), c=c)
    np.testing.assert_allclose(res, b)


def test_mobius_addition_zero_a(b, c):
    a = torch.zeros(100, 10, dtype=torch.float64)
    res = poincare.math.mobius_add(a, b, c=c)
    np.testing.assert_allclose(res, b)


def test_mobius_addition_zero_b(a, c):
    b = torch.zeros(100, 10, dtype=torch.float64)
    res = poincare.math.mobius_add(a, b, c=c)
    np.testing.assert_allclose(res, a)


def test_mobius_addition_negative_cancellation(a, c):
    res = poincare.math.mobius_add(a, -a, c=c)
    np.testing.assert_allclose(res, torch.zeros_like(res), atol=1e-10)


def test_mobius_negative_addition(a, b, c):
    res = poincare.math.mobius_add(-b, -a, c=c)
    res1 = -poincare.math.mobius_add(b, a, c=c)
    np.testing.assert_allclose(res, res1, atol=1e-10)


@pytest.mark.parametrize("n", list(range(5)))
def test_n_additions_via_scalar_multiplication(n, a, c):
    y = torch.zeros_like(a)
    for _ in range(n):
        y = poincare.math.mobius_add(a, y, c=c)
    ny = poincare.math.mobius_scalar_mul(n, a, c=c)
    np.testing.assert_allclose(y, ny, atol=1e-7, rtol=1e-10)


@pytest.fixture
def r1(seed):
    if seed % 3 == 0:
        return random.uniform(-1, 1)
    else:
        return torch.rand(100, 1, dtype=torch.float64) * 2 - 1


@pytest.fixture
def r2(seed):
    if seed % 3 == 1:
        return random.uniform(-1, 1)
    else:
        return torch.rand(100, 1, dtype=torch.float64) * 2 - 1


def test_scalar_multiplication_distributive(a, c, r1, r2):
    res = poincare.math.mobius_scalar_mul(r1 + r2, a, c=c)
    res1 = poincare.math.mobius_add(
        poincare.math.mobius_scalar_mul(r1, a, c=c),
        poincare.math.mobius_scalar_mul(r2, a, c=c),
        c=c,
    )
    res2 = poincare.math.mobius_add(
        poincare.math.mobius_scalar_mul(r1, a, c=c),
        poincare.math.mobius_scalar_mul(r2, a, c=c),
        c=c,
    )
    np.testing.assert_allclose(res1, res, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(res2, res, atol=1e-7, rtol=1e-10)


def test_scalar_multiplication_associative(a, c, r1, r2):
    res = poincare.math.mobius_scalar_mul(r1 * r2, a, c=c)
    res1 = poincare.math.mobius_scalar_mul(
        r1, poincare.math.mobius_scalar_mul(r2, a, c=c), c=c
    )
    res2 = poincare.math.mobius_scalar_mul(
        r2, poincare.math.mobius_scalar_mul(r1, a, c=c), c=c
    )
    np.testing.assert_allclose(res1, res, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(res2, res, atol=1e-7, rtol=1e-10)


def test_scaling_property(a, c, r1):
    x1 = a / a.norm(dim=-1, keepdim=True)
    ra = poincare.math.mobius_scalar_mul(r1, a, c=c)
    x2 = poincare.math.mobius_scalar_mul(abs(r1), a, c=c) / ra.norm(
        dim=-1, keepdim=True
    )
    np.testing.assert_allclose(x1, x2, atol=1e-7, rtol=1e-10)
