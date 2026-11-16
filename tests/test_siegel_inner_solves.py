import geoopt
import pytest
import torch


def symmetric(shape, dtype):
    raw = torch.randn(*shape, dtype=dtype)
    return (raw + raw.transpose(-1, -2)) / 2


def point(model, shape, dtype):
    raw = symmetric(shape, dtype)
    if model == "bounded":
        return raw / (2 * torch.linalg.matrix_norm(raw, keepdim=True))
    y = raw.imag @ raw.imag.transpose(-1, -2) + torch.eye(shape[-1])
    return torch.complex(raw.real, y)


def inverse_reference(model, z, u, v, keepdim):
    v = u if v is None else v
    if model == "bounded":
        identity = torch.eye(z.shape[-1], dtype=z.dtype)
        left = torch.linalg.inv(identity - z.conj() @ z)
        right = torch.linalg.inv(identity - z @ z.conj())
    else:
        left = right = torch.linalg.inv(z.imag).to(z.dtype)
    return torch.diagonal(left @ u @ right @ v.conj(), dim1=-2, dim2=-1).sum(
        -1, keepdim=keepdim
    )


@pytest.mark.parametrize("model", ["bounded", "upper"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "shapes", [((), (), ()), ((2,), (2,), (2,)), ((2, 1), (1, 3), ()), ((), (), (2, 3))]
)
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("keepdim", [False, True])
def test_siegel_inner_solve_values_and_broadcast(model, dtype, shapes, norm, keepdim):
    torch.manual_seed(1750)
    manifold = geoopt.BoundedDomain() if model == "bounded" else geoopt.UpperHalf()
    z = point(model, (*shapes[0], 3, 3), dtype)
    u = symmetric((*shapes[1], 3, 3), dtype)
    v = None if norm else symmetric((*shapes[2], 3, 3), dtype)
    expected = inverse_reference(model, z, u, v, keepdim)
    actual = manifold.inner(z, u, v, keepdim=keepdim)
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("model", ["bounded", "upper"])
@pytest.mark.parametrize("norm", [False, True])
def test_siegel_inner_gradients(model, norm):
    torch.manual_seed(1751)
    manifold = geoopt.BoundedDomain() if model == "bounded" else geoopt.UpperHalf()
    z = point(model, (2, 2), torch.complex128).requires_grad_()
    u = symmetric((2, 2), torch.complex128).requires_grad_()
    v = symmetric((2, 2), torch.complex128).requires_grad_()

    def actual(a, b, c):
        value = manifold.inner(a, b, None if norm else c)
        return value.real + 0.37 * value.imag

    assert torch.autograd.gradcheck(actual, (z, u, v))
    expected_value = inverse_reference(model, z, u, None if norm else v, False)
    expected = torch.autograd.grad(
        expected_value.real + 0.37 * expected_value.imag, (z, u) if norm else (z, u, v)
    )
    observed = torch.autograd.grad(actual(z, u, v), (z, u) if norm else (z, u, v))
    for gradient, reference in zip(observed, expected):
        torch.testing.assert_close(gradient, reference, rtol=1e-8, atol=1e-8)
