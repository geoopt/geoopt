import torch

import geoopt
import geoopt.layers


def assert_lorentz_point(manifold, point, *, atol=1e-5, rtol=1e-5):
    ok, reason = manifold._check_point_on_manifold(point, atol=atol, rtol=rtol)
    assert ok, reason


def test_lorentz_plfc_shape_and_manifold():
    dtype = torch.float64
    manifold = geoopt.Lorentz(k=torch.tensor(1.0, dtype=dtype))
    layer = geoopt.layers.LorentzPLFC(3, 2, manifold=manifold).to(dtype)
    input = manifold.random_normal(4, 3 + 1).data

    output = layer(input)

    assert output.shape == (4, 2 + 1)
    assert torch.isfinite(output).all()
    assert_lorentz_point(manifold, output, atol=1e-6, rtol=1e-6)


def test_lorentz_plfc_batch_dims():
    dtype = torch.float64
    manifold = geoopt.Lorentz(k=torch.tensor(2.0, dtype=dtype))
    layer = geoopt.layers.LorentzPLFC(4, 3, manifold=manifold, bias=False).to(dtype)
    input = manifold.random_normal(2, 5, 4 + 1).data

    output = layer(input)

    assert output.shape == (2, 5, 3 + 1)
    assert torch.isfinite(output).all()
    assert_lorentz_point(manifold, output, atol=1e-6, rtol=1e-6)


def test_lorentz_plfc_signed_distance_matches_output_space():
    dtype = torch.float64
    manifold = geoopt.Lorentz(k=torch.tensor(1.0, dtype=dtype))
    layer = geoopt.layers.LorentzPLFC(3, 2, manifold=manifold, bias=False).to(dtype)
    input = manifold.random_normal(4, 3 + 1).data

    distance = layer.signed_distance(input)
    output = layer(input)
    expected_space = torch.sqrt(manifold.k) * torch.sinh(distance / torch.sqrt(manifold.k))

    torch.testing.assert_close(output[..., 1:], expected_space)
    assert_lorentz_point(manifold, output, atol=1e-6, rtol=1e-6)


def test_lorentz_plfc_backward():
    dtype = torch.float64
    manifold = geoopt.Lorentz(k=torch.tensor(1.0, dtype=dtype))
    layer = geoopt.layers.LorentzPLFC(3, 2, manifold=manifold).to(dtype)
    input = manifold.random_normal(4, 3 + 1).data.detach().requires_grad_()

    output = layer(input)
    output.sum().backward()

    assert torch.isfinite(input.grad).all()
    assert torch.isfinite(layer.z.grad).all()
    assert torch.isfinite(layer.a.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
