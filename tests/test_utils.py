import pytest
import torch
import numpy as np
import geoopt


@pytest.fixture
def A():
    torch.manual_seed(42)
    n = 10
    a = torch.randn(n, 3, 3).double()
    a[:, 2, :] = 0
    return a.clone().requires_grad_()


def test_svd(A):
    u, d, v = geoopt.linalg.svd(A)
    with torch.no_grad():
        for i, a in enumerate(A):
            ut, dt, vt = torch.svd(a)
            np.testing.assert_allclose(u.detach()[i], ut.detach())
            np.testing.assert_allclose(d.detach()[i], dt.detach())
            np.testing.assert_allclose(v.detach()[i], vt.detach())
    u.sum().backward()  # this should work


def test_qr(A):
    q, r = geoopt.linalg.qr(A)
    with torch.no_grad():
        for i, a in enumerate(A):
            qt, rt = torch.qr(a)
            np.testing.assert_allclose(q.detach()[i], qt.detach())
            np.testing.assert_allclose(r.detach()[i], rt.detach())


def test_expm(A):
    from scipy.linalg import expm
    import numpy as np

    expm_scipy = np.zeros_like(A.detach())
    for i in range(A.shape[0]):
        expm_scipy[i] = expm(A.detach()[i].numpy())
    expm_torch = geoopt.linalg.expm(A)
    np.testing.assert_allclose(expm_torch.detach(), expm_scipy, rtol=1e-6)
    expm_torch.sum().backward()  # this should work
