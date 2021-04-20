import pytest
import geoopt


@pytest.mark.parametrize(
    "mf", [geoopt.PoincareBall, geoopt.SphereProjection, geoopt.Stereographic]
)
def test_wrapped_normal(mf):
    manifold = mf()
    mean = manifold.random_normal(2)
    point = manifold.wrapped_normal((4, 2), mean=mean)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold


def test_wrapped_normal_StereographicProductManifold():
    manifold = geoopt.StereographicProductManifold(
        (geoopt.PoincareBall(), 2),
        (geoopt.SphereProjection(), 2),
        (geoopt.Stereographic(), 2),
    )
    mean = manifold.random(6)
    point = manifold.wrapped_normal((4, 6), mean=mean)
    manifold.assert_check_point_on_manifold(point)
    assert point.manifold is manifold
