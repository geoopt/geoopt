import copy
import io
import pickle

import geoopt
import pytest
import torch


CASES = [
    ("Euclidean", "random", (4, 3)),
    ("Euclidean", "random_normal", (4, 3)),
    ("Sphere", "random", (4, 3)),
    ("Sphere", "random_uniform", (4, 3)),
    ("Stiefel", "random", (4, 3, 2)),
    ("Stiefel", "random_naive", (4, 3, 2)),
    ("BirkhoffPolytope", "random", (4, 3, 3)),
    ("BirkhoffPolytope", "random_naive", (4, 3, 3)),
    ("PoincareBall", "random", (4, 3)),
    ("PoincareBall", "random_normal", (4, 3)),
    ("PoincareBall", "wrapped_normal", (4, 3)),
    ("Lorentz", "random_normal", (4, 3)),
    ("ProductManifold", "random", (4, 5)),
    ("ProductManifold", "random_combined", (4, 5)),
]


def make_manifold(name):
    if name == "ProductManifold":
        return geoopt.ProductManifold(
            (geoopt.Sphere(), 3), (geoopt.Euclidean(ndim=1), 2)
        ).double()
    return getattr(geoopt, name)().double()


@pytest.mark.parametrize("name,method,shape", CASES)
@pytest.mark.parametrize("nested", [False, True])
def test_sample_belongs_to_scaled_manifold(name, method, shape, nested):
    base = make_manifold(name)
    manifold = geoopt.Scaled(base, 2).double()
    if nested:
        manifold = geoopt.Scaled(manifold, 3).double()
    kwargs = {"mean": torch.zeros(3, dtype=torch.float64)} if method == "wrapped_normal" else {}
    sample = getattr(manifold, method)(
        *shape,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(22),
        **kwargs,
    )

    assert sample.manifold is manifold
    assert sample.dtype == torch.float64
    assert sample.shape == shape
    manifold.assert_check_point_on_manifold(sample)
    assert geoopt.ManifoldParameter(sample).manifold is manifold
    assert geoopt.ManifoldParameter(sample, manifold=manifold).manifold is manifold


@pytest.mark.parametrize("nested", [False, True])
def test_optimizer_uses_the_sample_scale(nested):
    manifold = geoopt.Scaled(geoopt.Euclidean(ndim=1), 3).double()
    scale = 3
    if nested:
        manifold = geoopt.Scaled(manifold, 2).double()
        scale *= 2
    parameter = geoopt.ManifoldParameter(
        manifold.random(3, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    )
    initial = parameter.detach().clone()
    parameter.sum().backward()
    optimizer = geoopt.optim.RiemannianSGD([parameter], lr=0.3)
    optimizer.step()

    torch.testing.assert_close(parameter, initial - 0.3 / scale)


def test_sample_distance_keeps_the_scale():
    base = geoopt.Euclidean(ndim=1)
    manifold = geoopt.Scaled(base, 3).double()
    sample = manifold.random(3, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    target = torch.zeros_like(sample)

    torch.testing.assert_close(sample.dist(target, p="g"), 3 * base.dist(sample, target))


@pytest.mark.parametrize("name", ["Euclidean", "PoincareBall"])
def test_sampling_values_and_generator_are_preserved(name):
    base = make_manifold(name)
    manifold = geoopt.Scaled(base, 2).double()
    actual_generator = torch.Generator().manual_seed(23)
    expected_generator = torch.Generator().manual_seed(23)
    actual = manifold.random_normal(
        4, 3, std=0.8, dtype=torch.float64, generator=actual_generator
    )
    expected = base.random_normal(
        4, 3, std=0.4, dtype=torch.float64, generator=expected_generator
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
    assert expected.manifold is base


class CachedPointManifold(geoopt.Euclidean):
    def __init__(self):
        super().__init__(ndim=1)
        self.cached = geoopt.ManifoldTensor(
            torch.ones(3, dtype=torch.float64), manifold=self, requires_grad=True
        )

    def sample_point(self):
        return self.cached


def test_delegated_sampler_preserves_autograd_and_base_metadata():
    base = CachedPointManifold()
    manifold = geoopt.Scaled(base, 2).double()
    sample = manifold.sample_point()

    assert sample.manifold is manifold
    assert base.cached.manifold is base
    assert sample is not base.cached
    assert sample.data_ptr() == base.cached.data_ptr()
    sample.sum().backward()
    torch.testing.assert_close(base.cached.grad, torch.ones_like(base.cached))


def test_delegated_plain_tensor_method_is_unchanged():
    base = geoopt.Euclidean(ndim=1)
    manifold = geoopt.Scaled(base, 2)
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    torch.testing.assert_close(manifold.pairwise_inner(points, points), points @ points.T)


def test_delegated_classmethod_keeps_its_class_binding():
    sphere = geoopt.Sphere()
    euclidean = geoopt.Euclidean(ndim=1)
    base = geoopt.ProductManifold((sphere, 3), (euclidean, 2))
    manifold = geoopt.Scaled(base, 2)
    product = manifold.from_point(sphere.random(3), euclidean.random(2))

    assert isinstance(product, geoopt.ProductManifold)
    assert list(product.manifolds) == [sphere, euclidean]


def roundtrip(value, method):
    if method == "pickle":
        return pickle.loads(pickle.dumps(value))
    if method == "deepcopy":
        return copy.deepcopy(value)
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


@pytest.mark.parametrize("name", ["Euclidean", "Sphere", "PoincareBall"])
@pytest.mark.parametrize("method", ["pickle", "torch", "deepcopy"])
@pytest.mark.parametrize("parameter", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_sample_roundtrip_retains_scaled_geometry(name, method, parameter, nested):
    base = make_manifold(name)
    manifold = geoopt.Scaled(base, 3, learnable=True).double()
    if nested:
        manifold = geoopt.Scaled(manifold, 2).double()
    sample = manifold.random(3, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    if parameter:
        sample = geoopt.ManifoldParameter(sample)
    result = roundtrip(sample, method)

    assert type(result) is type(sample)
    assert isinstance(result.manifold, geoopt.Scaled)
    torch.testing.assert_close(result, sample)
    torch.testing.assert_close(result.manifold.scale, manifold.scale)
    target = manifold.random(3, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(result.dist(target, p="g"), manifold.dist(sample, target))
    tangent = manifold.proju(sample, torch.full_like(sample, 0.01))
    torch.testing.assert_close(result.retr(tangent), manifold.retr(sample, tangent))
    resumed = result.manifold.random(
        3, dtype=torch.float64, generator=torch.Generator().manual_seed(5)
    )
    torch.testing.assert_close(resumed, sample)
    assert resumed.manifold is result.manifold
    original_parameters = list(manifold.parameters())
    restored_parameters = list(result.manifold.parameters())
    assert len(restored_parameters) == len(original_parameters)
    for actual, expected in zip(restored_parameters, original_parameters):
        assert actual.requires_grad == expected.requires_grad
        torch.testing.assert_close(actual, expected)
    expected_gradients = torch.autograd.grad(
        manifold.dist(sample, target).sum(),
        [p for p in original_parameters if p.requires_grad],
    )
    actual_gradients = torch.autograd.grad(
        result.dist(target, p="g").sum(),
        [p for p in restored_parameters if p.requires_grad],
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    point = geoopt.ManifoldParameter(result)
    expected_step = manifold.retr(
        sample.detach(), -0.1 * manifold.egrad2rgrad(sample.detach(), torch.ones_like(sample))
    )
    point.sum().backward()
    geoopt.optim.RiemannianSGD([point], lr=0.1).step()
    torch.testing.assert_close(point, expected_step)


@pytest.mark.parametrize("method", ["pickle", "torch", "deepcopy"])
@pytest.mark.parametrize("nested", [False, True])
def test_roundtrip_preserves_shared_manifold_identity(method, nested):
    base = geoopt.PoincareBall(c=0.5, learnable=True).double()
    manifold = geoopt.Scaled(base, 3, learnable=True).double()
    if nested:
        manifold = geoopt.Scaled(manifold, 2, learnable=True).double()
    sample = manifold.random(3, generator=torch.Generator().manual_seed(5))
    parameter = geoopt.ManifoldParameter(sample)
    restored_base, restored_manifold, point, restored_parameter, repeated = roundtrip(
        (base, manifold, sample, parameter, sample), method
    )

    assert point is repeated
    assert point.manifold is restored_manifold
    assert restored_parameter.manifold is restored_manifold
    if nested:
        assert restored_manifold.base.base is restored_base
    else:
        assert restored_manifold.base is restored_base
    torch.testing.assert_close(restored_base.c, base.c)


@pytest.mark.parametrize("nested", [False, True])
def test_module_state_dict_checkpoint_is_preserved(nested):
    manifold = geoopt.Scaled(geoopt.Euclidean(ndim=1), 3).double()
    if nested:
        manifold = geoopt.Scaled(manifold, 2).double()
    module = torch.nn.Module()
    module.register_parameter("point", geoopt.ManifoldParameter(manifold.random(3)))
    expected = module.point.detach().clone()
    state = roundtrip(module.state_dict(), "torch")
    assert type(state["point"]) is torch.Tensor
    with torch.no_grad():
        module.point.zero_()
    module.load_state_dict(state)

    torch.testing.assert_close(module.point, expected)
