import pytest
import torch
import numpy as np
import geoopt


@pytest.fixture
def A():
    torch.manual_seed(42)
    n = 10
    a = torch.randn(n, 3, 3)
    a[:, 2, :] = 0
    return a


def test_svd(A):
    u, d, v = geoopt.linalg.batch_linalg.svd(A)
