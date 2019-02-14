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


def test_mobius_addition_left_cancelation_test(seed):
    a = torch.randn(100, 10, dtype=torch.float64)
    b = torch.randn(100, 10, dtype=torch.float64)

    if seed == 30:
        c = 0
    else:
        c = random.random()
    a = poincare.math.project(a, c=c)
    b = poincare.math.project(b, c=c)

    res = poincare.math.mobius_add(-a, poincare.math.mobius_add(a, b, c=c), c=c)
    np.testing.assert_allclose(res, b)


def test_mobius_addition_left_cancelation_test_broadcasted(seed):
    a = torch.randn(100, 10, dtype=torch.float64)
    b = torch.randn(100, 10, dtype=torch.float64)

    if seed == 30:
        c = torch.zeros(*a.shape[:-1], 1, dtype=torch.float64)
    else:
        c = torch.rand(*a.shape[:-1], 1, dtype=torch.float64)
    a = poincare.math.project(a, c=c)
    b = poincare.math.project(b, c=c)

    res = poincare.math.mobius_add(-a, poincare.math.mobius_add(a, b, c=c), c=c)
    np.testing.assert_allclose(res, b)
