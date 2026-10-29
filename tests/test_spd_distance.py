import numpy as np
import pytest
import scipy.linalg
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("case", ["isotropic_x", "isotropic_y", "both_isotropic", "coincident"])
def test_aim_repeated_eigenvalues_have_analytic_gradients(dtype, rotated, case):
    """Identity initialization and isotropic targets retain their distance derivatives."""
    values = torch.tensor([0.6, 1.2, 2.0], dtype=dtype)
    ones = torch.ones(3, dtype=dtype)
    x_values, y_values = {
        "isotropic_x": (ones, values),
        "isotropic_y": (values, 2 * ones),
        "both_isotropic": (ones, 2 * ones),
        "coincident": (ones, ones),
    }[case]
    q = torch.tensor([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    if not rotated:
        q = torch.eye(3, dtype=dtype)

    def matrix(eigenvalues):
        if torch.all(eigenvalues == eigenvalues[0]):
            return eigenvalues[0] * torch.eye(3, dtype=dtype)
        return q @ torch.diag(eigenvalues) @ q.T

    x = matrix(x_values).requires_grad_()
    y = matrix(y_values).requires_grad_()
    log_ratios = (y_values / x_values).log()
    loss = geoopt.SymmetricPositiveDefinite().dist(x, y).square()
    gradients = torch.autograd.grad(loss, (x, y))
    expected = (
        q @ torch.diag(-2 * log_ratios / x_values) @ q.T,
        q @ torch.diag(2 * log_ratios / y_values) @ q.T,
    )
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-10, 1e-12)
    torch.testing.assert_close(loss, log_ratios.square().sum(), rtol=rtol, atol=atol)
    for gradient, reference in zip(gradients, expected):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("keepdim", [False, True])
def test_aim_broadcast_matches_generalized_eigenvalues(dtype, keepdim):
    """Noncommuting covariances agree with an independent generalized eigenvalue solver."""
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(2, 1, 3, 3, generator=generator, dtype=dtype)
    b = torch.randn(3, 3, 3, generator=generator, dtype=dtype)
    x = a @ a.transpose(-1, -2) + torch.eye(3, dtype=dtype)
    y = b @ b.transpose(-1, -2) + torch.eye(3, dtype=dtype)
    actual = geoopt.SymmetricPositiveDefinite().dist(x, y, keepdim=keepdim)
    x_broadcast, y_broadcast = torch.broadcast_tensors(x, y)
    expected = np.empty((2, 3))
    for i in range(2):
        for j in range(3):
            eigenvalues = scipy.linalg.eigvalsh(
                y_broadcast[i, j].double().numpy(), x_broadcast[i, j].double().numpy()
            )
            expected[i, j] = np.linalg.norm(np.log(eigenvalues))
    if keepdim:
        expected = expected[..., None, None]
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype))


@pytest.mark.parametrize("case", ["isotropic_x", "isotropic_y", "both_isotropic", "noncommuting"])
def test_aim_gradcheck_in_symmetric_coordinates(case):
    """Finite differences use valid SPD perturbations, including repeated spectra."""
    p = torch.tensor([[0.3, 0.1], [0.1, -0.2]], dtype=torch.float64)
    q = torch.tensor([[-0.1, 0.2], [0.2, 0.4]], dtype=torch.float64)
    if case in ("isotropic_x", "both_isotropic"):
        p = torch.zeros_like(p)
    if case in ("isotropic_y", "both_isotropic"):
        q = 0.5 * torch.eye(2, dtype=torch.float64)
    p.requires_grad_()
    q.requires_grad_()
    manifold = geoopt.SymmetricPositiveDefinite()

    def squared_distance(left, right):
        x = torch.matrix_exp(0.5 * (left + left.T))
        y = torch.matrix_exp(0.5 * (right + right.T))
        return manifold.dist(x, y).square()

    assert torch.autograd.gradcheck(squared_distance, (p, q))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aim_optimizer_learns_from_identity(dtype):
    """A covariance parameter initialized at the manifold origin can take finite steps."""
    manifold = geoopt.SymmetricPositiveDefinite()
    point = geoopt.ManifoldParameter(torch.eye(3, dtype=dtype), manifold=manifold)
    target = torch.diag(torch.tensor([0.6, 1.2, 2.0], dtype=dtype))
    optimizer = geoopt.optim.RiemannianSGD([point], lr=0.05)
    initial = manifold.dist(point, target).square().detach()
    for _ in range(8):
        optimizer.zero_grad()
        loss = manifold.dist(point, target).square()
        loss.backward()
        assert torch.isfinite(point.grad).all()
        optimizer.step()
        assert torch.isfinite(point).all()
        assert manifold.check_point_on_manifold(point)
    assert manifold.dist(point, target).square() < initial
