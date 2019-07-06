import pytest
import geoopt


@pytest.mark.parametrize("ndim", [0, 1, 2])
def test_ndim_expected_behaviour(ndim):
    eucl = geoopt.Euclidean(ndim=ndim)
    point = eucl.random_normal(2, 2, 2, 2, 2)
    point1 = eucl.random_normal(2, 2, 2, 2, 2)
    # since spaces are the same we just use same method in test
    tangent = eucl.random_normal(2, 2, 2, 2, 2)

    inner = eucl.inner(point, tangent, tangent)
    assert inner.dim() == tangent.dim() - ndim

    inner = eucl.inner(point, tangent)
    assert inner.dim() == tangent.dim() - ndim

    norm = eucl.norm(point, tangent)
    assert norm.dim() == tangent.dim() - ndim

    dist = eucl.dist(point, point1)
    assert dist.dim() == point.dim() - ndim

    # keepdim now

    inner = eucl.inner(point, tangent, tangent, keepdim=True)
    assert inner.dim() == tangent.dim()

    inner = eucl.inner(point, tangent, keepdim=True)
    assert inner.dim() == tangent.dim()

    norm = eucl.norm(point, tangent, keepdim=True)
    assert norm.dim() == tangent.dim()

    dist = eucl.dist(point, point1, keepdim=True)
    assert dist.dim() == point.dim()
