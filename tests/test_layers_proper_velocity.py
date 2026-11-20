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


def test_gyro_pv_bn_normalizes_statistics():
    torch.manual_seed(0)
    manifold = make_manifold(1.0)
    layer = geoopt.layers.GyroProperVelocityBatchNorm(3, manifold=manifold, tol=1e-12).to(DTYPE)
    input = manifold.random_normal(64, 3, std=4.0, mean=1.0).detach()

    output = layer(input)
    mean = manifold.frechet_mean(output, tol=1e-12)
    var = manifold.frechet_variance(output, mean)

    torch.testing.assert_close(mean, torch.zeros(3, dtype=DTYPE), atol=1e-8, rtol=0)
    batch_var = manifold.frechet_variance(input, manifold.frechet_mean(input, tol=1e-12))
    torch.testing.assert_close(var, batch_var / (batch_var + layer.eps))


def test_gyro_pv_bn_bias_and_scale():
    torch.manual_seed(0)
    manifold = make_manifold(1.0)
    layer = geoopt.layers.GyroProperVelocityBatchNorm(3, manifold=manifold, tol=1e-12).to(DTYPE)
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([0.5, -1.0, 2.0], dtype=DTYPE))
        layer.log_scale.fill_(0.3)
    input = manifold.random_normal(64, 3, std=2.0).detach()

    output = layer(input)
    mean = manifold.frechet_mean(output, tol=1e-12)

    torch.testing.assert_close(mean, layer.bias.detach(), atol=1e-8, rtol=0)
    centered = manifold.gyroadd(-layer.bias.detach(), output)
    var0 = manifold.frechet_variance(centered, torch.zeros(3, dtype=DTYPE))
    batch_var = manifold.frechet_variance(input, manifold.frechet_mean(input, tol=1e-12))
    torch.testing.assert_close(
        var0, torch.exp(torch.tensor(0.6, dtype=DTYPE)) * batch_var / (batch_var + layer.eps)
    )


def test_gyro_pv_bn_running_stats_and_eval():
    manifold = make_manifold(1.0)
    layer = geoopt.layers.GyroProperVelocityBatchNorm(3, manifold=manifold, momentum=0.5).to(DTYPE)
    input = manifold.random_normal(32, 3, mean=1.0).detach()

    layer(input)
    assert not torch.allclose(layer.running_mean, torch.zeros(3, dtype=DTYPE))
    assert not torch.allclose(layer.running_var, torch.ones((), dtype=DTYPE))

    layer.eval()
    running_mean = layer.running_mean.clone()
    output = layer(input)
    expected = manifold.gyroscalar(
        torch.rsqrt(layer.running_var + layer.eps), manifold.gyroadd(-running_mean, input)
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(layer.running_mean, running_mean)


def test_gyro_pv_bn_batch_dims_affine_false_and_backward():
    manifold = make_manifold(2.0)
    layer = geoopt.layers.GyroProperVelocityBatchNorm(
        4, manifold=manifold, affine=False, track_running_stats=False
    ).to(DTYPE)
    input = manifold.random_normal(2, 6, 4).detach().requires_grad_()

    output = layer(input)
    output.sum().backward()

    assert output.shape == (2, 6, 4)
    assert layer.bias is None and layer.running_mean is None
    assert torch.isfinite(input.grad).all()
    with pytest.raises(ValueError, match="last dimension"):
        layer(torch.randn(4, 3, dtype=DTYPE))


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


def test_gyro_pv_bn_differentiable_stats():
    torch.manual_seed(0)
    manifold = make_manifold(1.0)
    input = manifold.random_normal(16, 3, std=2.0).detach()
    weight = torch.randn(16, 3, dtype=DTYPE)

    def loss_fn(x, layer):
        return (layer(x) * weight).sum()

    layer = geoopt.layers.GyroProperVelocityBatchNorm(
        3, manifold=manifold, track_running_stats=False, tol=1e-12, differentiable_stats=True
    ).to(DTYPE)
    x = input.clone().requires_grad_()
    loss_fn(x, layer).backward()
    # central finite differences of the full normalization, including the batch statistics
    eps = 1e-6
    numeric = torch.zeros_like(input)
    for i in range(input.size(0)):
        for j in range(input.size(1)):
            delta = torch.zeros_like(input)
            delta[i, j] = eps
            numeric[i, j] = (loss_fn(input + delta, layer) - loss_fn(input - delta, layer)) / (
                2 * eps
            )
    torch.testing.assert_close(x.grad, numeric, atol=1e-5, rtol=1e-5)

    detached = geoopt.layers.GyroProperVelocityBatchNorm(
        3, manifold=manifold, track_running_stats=False, tol=1e-12
    ).to(DTYPE)
    x = input.clone().requires_grad_()
    loss_fn(x, detached).backward()
    assert not torch.allclose(x.grad, numeric, atol=1e-3)


def test_gyro_pv_bn_single_point():
    manifold = make_manifold(1.0)
    layer = geoopt.layers.GyroProperVelocityBatchNorm(4, manifold=manifold).to(DTYPE)
    input = torch.randn(4, dtype=DTYPE)
    output = layer(input)
    torch.testing.assert_close(output, layer.bias.detach())
    torch.testing.assert_close(
        layer.running_mean,
        manifold.geodesic(torch.tensor(0.1, dtype=DTYPE), torch.zeros(4, dtype=DTYPE), input),
    )


def test_gyro_pv_bn_validates_arguments_and_repr():
    with pytest.raises(ValueError, match="num_features"):
        geoopt.layers.GyroProperVelocityBatchNorm(0)
    assert "num_features=3" in repr(geoopt.layers.GyroProperVelocityBatchNorm(3))


def lorentz_midpoint(values, weights, manifold):
    # weighted mean of the lifted points rescaled onto the hyperboloid, back to PV coordinates
    mean = weights @ pvmath.pv_to_lorentz(values, k=manifold.k)
    norm = torch.sqrt(mean[..., :1] ** 2 - (mean[..., 1:] ** 2).sum(dim=-1, keepdim=True))
    return pvmath.lorentz_to_pv(torch.sqrt(manifold.k) * mean / norm)


def randomize_attention(layer):
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if name in ("query_weight", "key_weight", "value_weight"):
                parameter.normal_(std=0.3)
        if hasattr(layer, "key_map"):
            layer.key_map.r.normal_(std=0.5)
            layer.raw_temperature.normal_()
            layer.scale.fill_(1.7)
        else:
            layer.feature_scale.fill_(1.3)
    return layer


def homomorphisms(weight):
    maps = []
    for generator in weight.detach():
        hom = geoopt.layers.PVHomomorphism(generator.size(-1)).to(DTYPE)
        with torch.no_grad():
            hom.weight.copy_(generator)
        maps.append(hom)
    return maps


def test_pv_homomorphism_is_gyro_homomorphism_and_isometry():
    torch.manual_seed(0)
    manifold = make_manifold(1.3)
    layer = geoopt.layers.PVHomomorphism(5).to(DTYPE)
    x, y = torch.randn(2, 7, 5, dtype=DTYPE) * 2
    t = torch.tensor(0.7, dtype=DTYPE)

    torch.testing.assert_close(layer(x), x)
    with torch.no_grad():
        layer.weight.normal_()
    rotation = layer.rotation
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(5, dtype=DTYPE))
    torch.testing.assert_close(torch.det(rotation), torch.tensor(1.0, dtype=DTYPE))
    torch.testing.assert_close(layer(manifold.gyroadd(x, y)), manifold.gyroadd(layer(x), layer(y)))
    torch.testing.assert_close(layer(manifold.gyroscalar(t, x)), manifold.gyroscalar(t, layer(x)))
    torch.testing.assert_close(manifold.dist(layer(x), layer(y)), manifold.dist(x, y))


def test_pv_hyperplane_key_matches_mlr_and_flat_limit():
    torch.manual_seed(0)
    manifold = make_manifold(0.8)
    key = geoopt.layers.PVHyperplaneKey(5, 3, manifold=manifold).to(DTYPE)
    assert torch.all(key.z.abs() <= (6 / (5 + 3)) ** 0.5) and torch.all(key.r == 0)
    mlr = randomize(geoopt.layers.ProperVelocityMLR(5, 3, manifold=manifold).to(DTYPE))
    with torch.no_grad():
        key.z.copy_(mlr.z)
        key.r.copy_(mlr.r)
    input = torch.randn(8, 5, dtype=DTYPE)
    torch.testing.assert_close(key(input), mlr(input))

    # k -> infinity gives the fully connected layer x -> Zx + b with b_j = -r_j ||z_j||
    flat = geoopt.layers.PVHyperplaneKey(5, 3, manifold=make_manifold(1e12)).to(DTYPE)
    with torch.no_grad():
        flat.z.copy_(mlr.z)
        flat.r.copy_(mlr.r)
    expected = input @ mlr.z.T - mlr.r * mlr.z.norm(dim=-1)
    torch.testing.assert_close(flat(input), expected.detach(), rtol=1e-5, atol=1e-5)


def test_pv_midpoint_is_lorentz_centroid_and_geodesic_midpoint():
    from geoopt.layers.proper_velocity import _midpoint

    torch.manual_seed(0)
    manifold = make_manifold(1.3)
    values = torch.randn(6, 4, dtype=DTYPE) * 2
    weights = torch.softmax(torch.randn(3, 6, dtype=DTYPE), dim=-1)
    torch.testing.assert_close(
        _midpoint(values, weights, manifold.k, 1e-15), lorentz_midpoint(values, weights, manifold)
    )
    half = torch.full((1, 2), 0.5, dtype=DTYPE)
    torch.testing.assert_close(
        _midpoint(values[:2], half, manifold.k, 1e-15)[0],
        manifold.geodesic(torch.tensor(0.5, dtype=DTYPE), values[0], values[1]),
    )


def test_pv_busemann_attention_matches_definition():
    # Busemann function as lim_t d(q, Exp_0(t u)) - t and midpoints through the hyperboloid
    torch.manual_seed(0)
    manifold = make_manifold(1.3)
    layer = geoopt.layers.PVBusemannAttention(4, 2, manifold=manifold, temperature=2.0).to(DTYPE)
    randomize_attention(layer)
    query = torch.randn(5, 4, dtype=DTYPE)
    context = torch.randn(9, 4, dtype=DTYPE) * 1.5

    q_maps = homomorphisms(layer.query_weight)
    v_maps = homomorphisms(layer.value_weight)
    keys = layer.key_map(context).detach().view(9, 2, 4)
    heads = []
    for head in range(2):
        direction = keys[:, head] / keys[:, head].norm(dim=-1, keepdim=True)
        far = manifold.expmap0(30.0 * direction)
        busemann = manifold.dist(q_maps[head](query)[:, None], far[None]) - 30.0
        logits = -layer.temperature[head].detach() * busemann / layer.scale.detach()
        weights = torch.softmax(logits, dim=-1)
        heads.append(lorentz_midpoint(v_maps[head](context), weights, manifold))
    uniform = torch.full((1, 2), 0.5, dtype=DTYPE)
    expected = lorentz_midpoint(torch.stack(heads, dim=-2), uniform, manifold).squeeze(-2)

    torch.testing.assert_close(layer(query, context), expected.detach())


@pytest.mark.parametrize("num_heads", [1, 2])
def test_pv_linear_attention_matches_quadratic_midpoint(num_heads):
    torch.manual_seed(0)
    manifold = make_manifold(0.9)
    layer = geoopt.layers.PVLinearAttention(4, num_heads, manifold=manifold, power=1.5)
    layer = randomize_attention(layer.to(DTYPE))
    query = torch.randn(6, 4, dtype=DTYPE)
    context = torch.randn(11, 4, dtype=DTYPE) * 1.5

    q_maps = homomorphisms(layer.query_weight)
    k_maps = homomorphisms(layer.key_weight)
    v_maps = homomorphisms(layer.value_weight)
    heads = []
    for head in range(num_heads):
        weights = layer.feature_map(q_maps[head](query)) @ layer.feature_map(k_maps[head](context)).T
        weights = weights / weights.sum(dim=-1, keepdim=True)
        heads.append(lorentz_midpoint(v_maps[head](context), weights, manifold))
    if num_heads == 1:
        expected = heads[0]
    else:
        tangent = torch.cat([manifold.logmap0(output) for output in heads], dim=-1)
        expected = manifold.expmap0(layer.merge(tangent))

    torch.testing.assert_close(layer(query, context), expected.detach())


def test_pv_linear_attention_query_with_nonpositive_coordinates():
    # the feature map of such a query is ~eps, but its normalized weights are well defined
    torch.manual_seed(0)
    manifold = make_manifold(1.0)
    layer = geoopt.layers.PVLinearAttention(4, manifold=manifold).to(DTYPE)
    query = -torch.rand(3, 4, dtype=DTYPE) - 0.1
    context = torch.randn(10, 4, dtype=DTYPE)

    weights = layer.feature_map(query) @ layer.feature_map(context).T
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = lorentz_midpoint(context, weights, manifold).detach()

    torch.testing.assert_close(layer(query, context), expected)
    assert torch.all(expected.norm(dim=-1) > 1e-3)


@pytest.mark.parametrize("num_heads", [1, 3])
def test_pv_attention_flat_limit_is_euclidean_attention(num_heads):
    torch.manual_seed(0)
    manifold = make_manifold(1e12)
    query = torch.randn(5, 4, dtype=DTYPE)
    context = torch.randn(8, 4, dtype=DTYPE)

    busemann = geoopt.layers.PVBusemannAttention(4, num_heads, manifold=manifold).to(DTYPE)
    busemann = randomize_attention(busemann)
    keys = busemann.key_map(context).view(8, num_heads, 4).transpose(0, 1)
    direction = keys / keys.norm(dim=-1, keepdim=True)
    q = torch.stack([m(query) for m in homomorphisms(busemann.query_weight)])
    v = torch.stack([m(context) for m in homomorphisms(busemann.value_weight)])
    logits = busemann.temperature * (q @ direction.transpose(-1, -2)) / busemann.scale
    expected = (torch.softmax(logits, dim=-1) @ v).mean(dim=0)
    torch.testing.assert_close(busemann(query, context), expected.detach(), rtol=1e-5, atol=1e-5)

    linear = randomize_attention(geoopt.layers.PVLinearAttention(4, manifold=manifold).to(DTYPE))
    q_map, k_map, v_map = [homomorphisms(w)[0] for w in (linear.query_weight, linear.key_weight, linear.value_weight)]
    weights = linear.feature_map(q_map(query)) @ linear.feature_map(k_map(context)).T
    expected = weights @ v_map(context) / weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(linear(query, context), expected.detach(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cls", [geoopt.layers.PVBusemannAttention, geoopt.layers.PVLinearAttention])
def test_pv_attention_batch_dims_mask_and_backward(cls):
    torch.manual_seed(0)
    manifold = make_manifold(2.0)
    layer = randomize_attention(cls(4, 2, manifold=manifold).to(DTYPE))
    query = torch.randn(3, 5, 4, dtype=DTYPE, requires_grad=True)
    context = torch.randn(3, 7, 4, dtype=DTYPE)

    output = layer(query, context)
    assert output.shape == (3, 5, 4)
    torch.testing.assert_close(output[1], layer(query[1], context[1]))
    assert layer(query).shape == (3, 5, 4)

    keep = torch.tensor([True, False, True, True, False, True, True])
    torch.testing.assert_close(layer(query, context, mask=keep), layer(query, context[:, keep]))

    output.sum().backward()
    assert torch.isfinite(query.grad).all()
    for parameter in layer.parameters():
        if not parameter.requires_grad:  # the manifold curvature
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_pv_busemann_attention_float_mask_and_score_clip():
    torch.manual_seed(0)
    manifold = make_manifold(1.0)
    layer = randomize_attention(geoopt.layers.PVBusemannAttention(4, manifold=manifold).to(DTYPE))
    query, context = torch.randn(5, 4, dtype=DTYPE), torch.randn(6, 4, dtype=DTYPE)
    keep = torch.rand(5, 6) > 0.3
    keep[:, 0] = True
    additive = torch.zeros(5, 6, dtype=DTYPE).masked_fill(~keep, float("-inf"))
    torch.testing.assert_close(layer(query, context, mask=keep), layer(query, context, mask=additive))

    clipped = geoopt.layers.PVBusemannAttention(4, manifold=manifold, score_clip=1e-12).to(DTYPE)
    uniform = torch.full((5, 6), 1 / 6, dtype=DTYPE)
    torch.testing.assert_close(clipped(query, context), lorentz_midpoint(context, uniform, manifold))


def test_pv_attention_validates_arguments_and_repr():
    for cls in [geoopt.layers.PVBusemannAttention, geoopt.layers.PVLinearAttention]:
        with pytest.raises(ValueError, match="features"):
            cls(0)
        with pytest.raises(ValueError, match="num_heads"):
            cls(4, 0)
        with pytest.raises(ValueError, match="last dimension"):
            cls(4)(torch.randn(5, 3))
        assert "num_heads=2" in repr(cls(4, 2))
    with pytest.raises(ValueError, match="temperature"):
        geoopt.layers.PVBusemannAttention(4, temperature=0.0)
    with pytest.raises(ValueError, match="power"):
        geoopt.layers.PVLinearAttention(4, power=0.0)
    with pytest.raises(ValueError, match="features"):
        geoopt.layers.PVHomomorphism(0)
    with pytest.raises(ValueError, match="last dimension"):
        geoopt.layers.PVHomomorphism(3)(torch.randn(2, 4))
    assert "features=3" in repr(geoopt.layers.PVHomomorphism(3))
    assert geoopt.layers.PVLinearAttention(4).merge is None
