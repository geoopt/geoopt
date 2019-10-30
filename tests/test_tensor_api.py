import geoopt
import pytest


def test_allow_empty_parameter_compat():
    p = geoopt.ManifoldParameter()
    assert p.shape == (0,)


def test_compare_manifolds():
    m1 = geoopt.Euclidean()
    m2 = geoopt.Euclidean(ndim=1)
    tensor = geoopt.ManifoldTensor(10, manifold=m1)
    with pytest.raises(ValueError) as e:
        _ = geoopt.ManifoldParameter(tensor, manifold=m2)
    assert e.match("Manifolds do not match")
