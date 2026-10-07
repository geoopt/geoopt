"""Explicit RNG streams must survive every built-in sampling entry point."""

import pytest
import torch
import geoopt


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
]


@pytest.fixture(autouse=True)
def full_precision_sampling():
    # Membership tolerances require full float32 precision, not TF32 matmuls.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


CASES = [
    ("Euclidean", "random_normal", (4, 3)),
    ("Sphere", "random_uniform", (4, 3)),
    ("SphereExact", "random", (4, 3)),
    ("projected_sphere", "random", (4, 3)),
    ("Stiefel", "random_naive", (4, 3, 2)),
    ("EuclideanStiefelExact", "random", (4, 3, 2)),
    ("BirkhoffPolytope", "random_naive", (4, 3, 3)),
    ("SymmetricPositiveDefinite", "random", (4, 3, 3)),
    ("Lorentz", "random_normal", (4, 3)),
    ("PoincareBall", "random_normal", (4, 3)),
    ("PoincareBallExact", "random", (4, 3)),
    ("SphereProjection", "random_normal", (4, 3)),
    ("SphereProjectionExact", "random", (4, 3)),
    ("Stereographic", "random_normal", (4, 3)),
    ("StereographicExact", "random", (4, 3)),
    ("PoincareBall", "wrapped_normal", (4, 3)),
    ("SphereProjection", "wrapped_normal", (4, 3)),
    ("Stereographic", "wrapped_normal", (4, 3)),
    ("UpperHalf", "random", (4, 3, 3)),
    ("BoundedDomain", "random", (4, 3, 3)),
]


def make_case(name, method, precision, device):
    dtype = (torch.float32, torch.float64)[precision]
    if name == "projected_sphere":
        manifold = geoopt.Sphere(intersection=torch.eye(3, dtype=dtype)[:, :2])
    else:
        manifold = getattr(geoopt, name)()
    manifold = manifold.to(device=device, dtype=dtype)
    if name in ("UpperHalf", "BoundedDomain"):
        dtype = (torch.complex64, torch.complex128)[precision]
    # Match parameter/buffer devices exactly (cuda:0 rather than unindexed cuda).
    sampling_device = (
        torch.device("cuda", torch.cuda.current_device())
        if device == "cuda"
        else torch.device(device)
    )
    kwargs = dict(dtype=dtype, device=sampling_device)
    if method == "wrapped_normal":
        kwargs.update(mean=torch.full((3,), 0.1, dtype=dtype, device=device), std=0.2)
    return manifold, kwargs


def rng_states():
    return [torch.random.get_rng_state()] + (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )


def assert_states_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert torch.equal(a, b)


@pytest.fixture
def gaussian_draws(monkeypatch):
    draws = []
    randn = torch.randn
    normal_ = torch.Tensor.normal_

    def record_randn(*args, **kwargs):
        result = randn(*args, **kwargs)
        draws.append(result.clone())
        return result

    def record_normal(tensor, *args, **kwargs):
        result = normal_(tensor, *args, **kwargs)
        draws.append(result.clone())
        return result

    monkeypatch.setattr(torch, "randn", record_randn)
    monkeypatch.setattr(torch.Tensor, "normal_", record_normal)
    return draws


@pytest.mark.parametrize("name,method,shape", CASES)
@pytest.mark.parametrize("precision", [0, 1])
@pytest.mark.parametrize("device", DEVICES)
def test_generator_replay_and_isolation(name, method, shape, precision, device, gaussian_draws):
    manifold, kwargs = make_case(name, method, precision, device)
    sample = getattr(manifold, method)
    generator = torch.Generator(device=device).manual_seed(3407)
    initial = generator.get_state()
    global_states = rng_states()
    first = sample(*shape, generator=generator, **kwargs)
    after_first = generator.get_state()
    second = sample(shape, generator=generator, **kwargs)
    after_second = generator.get_state()
    assert_states_equal(global_states, rng_states())
    assert not torch.equal(initial, after_first)
    assert not torch.equal(after_first, after_second)
    expected_draws = list(gaussian_draws)
    assert expected_draws
    # Interleaved default-generator draws must not affect replay.
    torch.randn(17, device=device)
    gaussian_draws.clear()
    generator.set_state(initial)
    # Scripted transforms may round differently before and after JIT optimization.
    torch.testing.assert_close(sample(*shape, generator=generator, **kwargs), first)
    assert torch.equal(generator.get_state(), after_first)
    torch.testing.assert_close(sample(shape, generator=generator, **kwargs), second)
    assert torch.equal(generator.get_state(), after_second)
    assert len(gaussian_draws) == len(expected_draws)
    for actual, expected in zip(gaussian_draws, expected_draws):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert first.shape == shape
    assert first.dtype == kwargs["dtype"]
    assert first.device.type == device
    manifold.assert_check_point_on_manifold(first, atol=1e-4, rtol=1e-4)
    if isinstance(first, geoopt.ManifoldTensor):
        assert first.manifold is manifold


@pytest.mark.parametrize("name,method,shape", CASES)
def test_default_generator_unchanged(name, method, shape):
    manifold, kwargs = make_case(name, method, 1, "cpu")
    sample = getattr(manifold, method)
    with torch.random.fork_rng():
        torch.manual_seed(3407)
        expected = sample(*shape, **kwargs)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(3407)
        actual = sample(*shape, generator=None, **kwargs)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(torch.random.get_rng_state(), expected_state)


@pytest.mark.parametrize("name,method,shape", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_reference_gaussian_transform(name, method, shape, device):
    manifold, kwargs = make_case(name, method, 1, device)
    generator = torch.Generator(device=device).manual_seed(3407)
    reference = torch.Generator(device=device).manual_seed(3407)
    if name == "Euclidean":
        kwargs.update(mean=0.4, std=0.7)
    actual = getattr(manifold, method)(*shape, generator=generator, **kwargs)
    noise = torch.empty(shape, dtype=kwargs["dtype"], device=device).normal_(generator=reference)
    if name == "Euclidean":
        expected = noise * 0.7 + 0.4
    elif name in ("Sphere", "SphereExact", "projected_sphere"):
        if name == "projected_sphere":
            noise[..., 2] = 0
        expected = noise / noise.norm(dim=-1, keepdim=True)
    elif name in ("Stiefel", "EuclideanStiefelExact"):
        expected = torch.linalg.qr(noise).Q
    elif name == "BirkhoffPolytope":
        expected = manifold.projx(noise.abs())
    elif name == "SymmetricPositiveDefinite":
        expected = torch.matrix_exp((noise + noise.transpose(-1, -2)) / 4)
    elif name in ("UpperHalf", "BoundedDomain"):
        symmetric = (noise + noise.transpose(-1, -2)) / 4
        upper = torch.complex(symmetric.real, torch.matrix_exp(symmetric.imag))
        if name == "UpperHalf":
            expected = upper
        else:
            identity = torch.eye(shape[-1], dtype=upper.dtype, device=device)
            expected = torch.linalg.solve(upper + 1j * identity, upper - 1j * identity)
    elif name == "Lorentz":
        expected = manifold.expmap0(noise / noise.norm(dim=-1, keepdim=True))
    elif method == "wrapped_normal":
        # Conformal metric g_mu = lambda_mu^2 I, Algorithm 1 of Mathieu et al.
        mean = kwargs["mean"]
        conformal_factor = 2 / (1 + manifold.k * mean.square().sum())
        expected = manifold.expmap(mean, noise * kwargs["std"] / conformal_factor)
    else:
        expected = manifold.expmap0(noise / shape[-1] ** 0.5)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(generator.get_state(), reference.get_state())


@pytest.mark.parametrize("name,method,shape", CASES[:15])
def test_random_alias(name, method, shape):
    if name == "Lorentz":
        pytest.skip("Lorentz.random is not implemented upstream")
    manifold, kwargs = make_case(name, method, 1, "cpu")
    g1 = torch.Generator().manual_seed(22)
    g2 = torch.Generator().manual_seed(22)
    torch.testing.assert_close(
        getattr(manifold, method)(*shape, generator=g1, **kwargs),
        manifold.random(*shape, generator=g2, **kwargs),
    )
    assert torch.equal(g1.get_state(), g2.get_state())


@pytest.mark.parametrize("name,method,shape", [CASES[0], CASES[1], CASES[8], CASES[9], CASES[15]])
def test_scaled_generator(name, method, shape):
    base, kwargs = make_case(name, method, 1, "cpu")
    scaled = geoopt.Scaled(base, 2)
    g1 = torch.Generator().manual_seed(22)
    g2 = torch.Generator().manual_seed(22)
    expected_kwargs = dict(kwargs)
    if method in ("random_normal", "wrapped_normal"):
        expected_kwargs["std"] = kwargs.get("std", 1) / 2
    torch.testing.assert_close(
        getattr(scaled, method)(*shape, generator=g1, **kwargs),
        getattr(base, method)(*shape, generator=g2, **expected_kwargs),
    )
    assert torch.equal(g1.get_state(), g2.get_state())


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_product_generator(nested, device):
    sphere = geoopt.Sphere()
    euclidean = geoopt.Euclidean(ndim=1)
    if nested:
        first = geoopt.ProductManifold((sphere, 3), (euclidean, 2))
        first_size = 5
    else:
        first, first_size = sphere, 3
    product = geoopt.ProductManifold((first, first_size), (euclidean, 2)).to(device)
    g1 = torch.Generator(device=device).manual_seed(22)
    g2 = torch.Generator(device=device).manual_seed(22)
    states = rng_states()
    actual = product.random(4, product.n_elements, device=device, generator=g1)
    parts = [
        m.random((4,) + shape, device=device, generator=g2)
        for m, shape in zip(product.manifolds, product.shapes)
    ]
    torch.testing.assert_close(actual, product.pack_point(*parts))
    assert torch.equal(g1.get_state(), g2.get_state())
    assert_states_equal(states, rng_states())
    assert actual.manifold is product


@pytest.mark.parametrize("device", DEVICES)
def test_product_wrapped_generator(device):
    product = geoopt.StereographicProductManifold(
        (geoopt.PoincareBall(), 2), (geoopt.SphereProjection(), 3)
    ).to(device=device, dtype=torch.float64)
    mean = torch.full((5,), 0.1, dtype=torch.float64, device=device)
    std = torch.linspace(0.1, 0.3, 5, dtype=torch.float64, device=device)
    g1 = torch.Generator(device=device).manual_seed(22)
    g2 = torch.Generator(device=device).manual_seed(22)
    states = rng_states()
    actual = product.wrapped_normal(4, 5, mean=mean, std=std, generator=g1)
    parts = [
        m.wrapped_normal(
            4,
            *shape,
            mean=product.take_submanifold_value(mean, i),
            std=product.take_submanifold_value(std, i),
            generator=g2,
        )
        for i, (m, shape) in enumerate(zip(product.manifolds, product.shapes))
    ]
    torch.testing.assert_close(actual, product.pack_point(*parts))
    assert torch.equal(g1.get_state(), g2.get_state())
    assert_states_equal(states, rng_states())
    product.assert_check_point_on_manifold(actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name,method,shape", CASES)
def test_generator_device_mismatch(name, method, shape):
    manifold, kwargs = make_case(name, method, 1, "cuda")
    # Omit device: the projector/curvature device still governs those samplers.
    if name in ("projected_sphere", "Lorentz", "PoincareBall", "SphereProjection", "Stereographic"):
        kwargs.pop("device")
    with pytest.raises(RuntimeError, match="[Gg]enerator"):
        getattr(manifold, method)(*shape, generator=torch.Generator(), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,method,shape", [CASES[3], CASES[8], CASES[9], CASES[15]])
def test_inferred_sampling_device(name, method, shape, device):
    manifold, kwargs = make_case(name, method, 1, device)
    g1 = torch.Generator(device=device).manual_seed(22)
    g2 = torch.Generator(device=device).manual_seed(22)
    explicit = getattr(manifold, method)(*shape, generator=g1, **kwargs)
    kwargs.pop("device")
    kwargs.pop("dtype")
    inferred = getattr(manifold, method)(*shape, generator=g2, **kwargs)
    torch.testing.assert_close(inferred, explicit)
    assert inferred.dtype == torch.float64
    assert inferred.device.type == device
    assert torch.equal(g1.get_state(), g2.get_state())


@pytest.mark.parametrize("name", ["Sphere", "Euclidean", "PoincareBall"])
def test_scaled_random_alias_and_nested_wrapper(name):
    base = getattr(geoopt, name)().double()
    # Nested scales cancel, including signature-bound random aliases.
    scaled = geoopt.Scaled(geoopt.Scaled(base, 2), 0.5)
    g1 = torch.Generator().manual_seed(22)
    g2 = torch.Generator().manual_seed(22)
    states = rng_states()
    torch.testing.assert_close(
        scaled.random(4, 3, dtype=torch.float64, generator=g1),
        base.random(4, 3, dtype=torch.float64, generator=g2),
    )
    assert torch.equal(g1.get_state(), g2.get_state())
    assert_states_equal(states, rng_states())


def test_scaled_product_generator():
    product = geoopt.ProductManifold((geoopt.Sphere(), 3), (geoopt.Euclidean(), 2))
    scaled = geoopt.Scaled(product, 2)
    g1 = torch.Generator().manual_seed(22)
    g2 = torch.Generator().manual_seed(22)
    states = rng_states()
    torch.testing.assert_close(
        scaled.random(4, 5, generator=g1),
        product.random(4, 5, generator=g2),
    )
    assert torch.equal(g1.get_state(), g2.get_state())
    assert_states_equal(states, rng_states())
