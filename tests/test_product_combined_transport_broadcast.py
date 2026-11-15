import itertools

import pytest
import torch

import geoopt


def product():
    return geoopt.ProductManifold(
        (geoopt.Euclidean(ndim=1), 2),
        (geoopt.Sphere(), 3),
        (geoopt.Euclidean(ndim=2), (2, 2)),
        (geoopt.Euclidean(), ()),
    )


def separate_results(manifold, operation, x, u, v):
    x, u, v = torch.broadcast_tensors(x, u, v)
    points, vectors = [], []
    for index in itertools.product(*(range(n) for n in x.shape[:-1])):
        points.append(getattr(manifold, operation)(x[index], u[index]))
        vectors.append(
            getattr(manifold, "transp_follow_" + operation)(x[index], u[index], v[index])
        )
    return torch.stack(points).reshape(x.shape), torch.stack(vectors).reshape(v.shape)


@pytest.mark.parametrize("operation", ["retr", "expmap"])
@pytest.mark.parametrize(
    "point_shape,direction_shape,vector_shape",
    [
        ((), (), (2,)),
        ((3,), (1,), (2, 1)),
        ((1, 3), (2, 1), (4, 1, 1)),
        ((2, 3), (2, 3), (2, 3)),
    ],
    ids=["one-vector-batch", "two-batch-axes", "three-batch-axes", "equal-batches"],
)
def test_combined_transport_matches_individual_maps(
    operation, point_shape, direction_shape, vector_shape
):
    manifold = product()
    generator = torch.Generator().manual_seed(1729)

    def sample(shape):
        return torch.randn(*shape, manifold.n_elements, dtype=torch.float64, generator=generator)

    x = manifold.projx(sample(point_shape))
    u = manifold.proju(x, sample(direction_shape)) * 0.05
    v = manifold.proju(x, sample(vector_shape))
    original = [tensor.clone() for tensor in (x, u, v)]
    expected_point, expected_vector = separate_results(manifold, operation, x, u, v)

    point, vector = getattr(manifold, operation + "_transp")(x, u, v)

    torch.testing.assert_close(point, expected_point)
    torch.testing.assert_close(vector, expected_vector)
    manifold.assert_check_point_on_manifold(point)
    manifold.assert_check_vector_on_tangent(point, vector)
    for actual, saved in zip((x, u, v), original):
        torch.testing.assert_close(actual, saved)


@pytest.mark.parametrize("operation", ["retr", "expmap"])
def test_combined_transport_accumulates_broadcast_gradients(operation):
    manifold = geoopt.ProductManifold(
        (geoopt.Euclidean(ndim=1), 2),
        (geoopt.Euclidean(ndim=2), (2, 2)),
        (geoopt.Euclidean(), ()),
    )
    x = torch.arange(7, dtype=torch.float64, requires_grad=True)
    u = torch.linspace(0.1, 0.7, 7, dtype=torch.float64, requires_grad=True)
    v = torch.ones(3, 7, dtype=torch.float64, requires_grad=True)
    weights = torch.arange(21, dtype=torch.float64).reshape(3, 7)

    point, vector = getattr(manifold, operation + "_transp")(x, u, v)
    loss = (weights * point).sum() + (weights * vector).sum()
    gradients = torch.autograd.grad(loss, (x, u, v))

    torch.testing.assert_close(point, (x + u).expand_as(v))
    torch.testing.assert_close(vector, v)
    torch.testing.assert_close(gradients[0], weights.sum(0))
    torch.testing.assert_close(gradients[1], weights.sum(0))
    torch.testing.assert_close(gradients[2], weights)


@pytest.mark.parametrize("operation", ["retr", "expmap"])
def test_nested_product_accepts_batched_vectors(operation):
    inner = geoopt.ProductManifold((geoopt.Euclidean(ndim=1), 2), (geoopt.Sphere(), 3))
    manifold = geoopt.ProductManifold((inner, 5), (geoopt.Euclidean(), ()))
    x = torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0, 3.0], dtype=torch.float64)
    u = torch.tensor([0.1, 0.2, 0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    v = manifold.proju(x, torch.arange(18, dtype=torch.float64).reshape(3, 6))
    expected = separate_results(manifold, operation, x, u, v)

    actual = getattr(manifold, operation + "_transp")(x, u, v)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("operation", ["retr", "expmap"])
def test_unbatched_combined_transport_control(operation):
    manifold = product()
    x = manifold.projx(torch.arange(1, 11, dtype=torch.float64))
    u = manifold.proju(x, torch.ones_like(x)) * 0.05
    v = manifold.proju(x, torch.arange(10, dtype=torch.float64))

    point, vector = getattr(manifold, operation + "_transp")(x, u, v)

    torch.testing.assert_close(point, getattr(manifold, operation)(x, u))
    torch.testing.assert_close(vector, getattr(manifold, "transp_follow_" + operation)(x, u, v))
