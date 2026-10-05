import pytest
import torch

import geoopt
import geoopt.layers
from geoopt.manifolds.proper_velocity import math as pvmath

DTYPE = torch.float64


def make_manifold(k=1.0):
    return geoopt.ProperVelocity(k=torch.tensor(k, dtype=DTYPE))


def randomize(layer):
    with torch.no_grad():
        layer.z.normal_(std=0.5)
        layer.r.normal_()
        if getattr(layer, "bias", None) is not None:
            layer.bias.normal_(std=0.5)
    return layer


def test_pv_mlr_shape_and_batch_dims():
    manifold = make_manifold(2.0)
    layer = geoopt.layers.ProperVelocityMLR(4, 3, manifold=manifold).to(DTYPE)
    input = manifold.random_normal(2, 5, 4).detach()

    output = layer(input)

    assert output.shape == (2, 5, 3)
    assert torch.isfinite(output).all()


def test_pv_mlr_matches_point_to_hyperplane_distance():
    # v_j(x) = sqrt(k) ||z_j|| arsinh(<(-p_j) ⊕ x, z_j> / (sqrt(k) ||z_j||)), p_j = Exp_0(r_j z_j / ||z_j||)
    torch.manual_seed(0)
    manifold = make_manifold(0.7)
    layer = randomize(geoopt.layers.ProperVelocityMLR(5, 4, manifold=manifold).to(DTYPE))
    input = torch.randn(16, 5, dtype=DTYPE) * 2
    sqrt_k = torch.sqrt(manifold.k)

    z_norm = layer.z.norm(dim=-1)
    p = manifold.expmap0(layer.r[:, None] * layer.z / z_norm[:, None])
    shifted = manifold.gyroadd(-p, input[:, None, :])
    expected = sqrt_k * z_norm * torch.asinh((shifted * layer.z).sum(-1) / (sqrt_k * z_norm))

    torch.testing.assert_close(layer(input), expected)


def test_pv_layers_match_lorentz_plfc():
    # The PV model is the space-like part of the hyperboloid, so PV MLR / FC agree with
    # LorentzPLFC applied to the lifted points
    torch.manual_seed(0)
    k = 1.3
    manifold = make_manifold(k)
    lorentz = geoopt.Lorentz(k=torch.tensor(k, dtype=DTYPE))
    fc = randomize(geoopt.layers.ProperVelocityFC(5, 4, manifold=manifold).to(DTYPE))
    plfc = geoopt.layers.LorentzPLFC(5, 4, manifold=lorentz).to(DTYPE)
    with torch.no_grad():
        plfc.z.copy_(fc.z)
        plfc.a.copy_(fc.r)
        plfc.bias.copy_(fc.bias)
    input = torch.randn(16, 5, dtype=DTYPE) * 2
    lifted = pvmath.pv_to_lorentz(input, k=manifold.k)

    torch.testing.assert_close(fc.signed_distance(input), plfc.signed_distance(lifted))
    torch.testing.assert_close(fc(input), plfc(lifted)[..., 1:])


def test_pv_fc_output_and_bias():
    torch.manual_seed(0)
    manifold = make_manifold(1.0)
    layer = randomize(geoopt.layers.ProperVelocityFC(3, 2, manifold=manifold).to(DTYPE))
    input = torch.randn(8, 3, dtype=DTYPE)

    distance = layer.signed_distance(input)
    expected = torch.sqrt(manifold.k) * torch.sinh(distance / torch.sqrt(manifold.k))
    expected = manifold.gyroadd(expected, manifold.expmap0(layer.bias))

    torch.testing.assert_close(layer(input), expected)

    no_bias = geoopt.layers.ProperVelocityFC(3, 2, manifold=manifold, bias=False).to(DTYPE)
    assert no_bias.bias is None
    assert no_bias(input).shape == (8, 2)


def test_pv_fc_backward():
    manifold = make_manifold(1.0)
    layer = geoopt.layers.ProperVelocityFC(3, 2, manifold=manifold).to(DTYPE)
    input = manifold.random_normal(4, 3).detach().requires_grad_()

    layer(input).sum().backward()

    assert torch.isfinite(input.grad).all()
    for parameter in layer.parameters():
        if not parameter.requires_grad:  # the manifold curvature
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("cls", [geoopt.layers.ProperVelocityMLR, geoopt.layers.ProperVelocityFC])
def test_pv_layers_reject_wrong_input_shape(cls):
    layer = cls(3, 2, manifold=make_manifold()).to(DTYPE)
    with pytest.raises(ValueError, match="last dimension"):
        layer(torch.randn(4, 4, dtype=DTYPE))


def test_pv_layers_validate_arguments_and_repr():
    for cls in [geoopt.layers.ProperVelocityMLR, geoopt.layers.ProperVelocityFC]:
        with pytest.raises(ValueError, match="in_features"):
            cls(0, 2)
        with pytest.raises(ValueError, match="out_features"):
            cls(2, 0)

    fc = geoopt.layers.ProperVelocityFC(3, 2)
    with torch.no_grad():
        fc.bias.fill_(1.0)
    fc.reset_parameters()
    assert torch.all(fc.bias == 0) and torch.all(fc.r == 0)
    assert "bias=True" in repr(fc)
    assert "out_features=2" in repr(geoopt.layers.ProperVelocityMLR(3, 2))
