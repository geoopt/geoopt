from geoopt import ProductManifold, Sphere


def test_init():
    pman = ProductManifold((Sphere(), 10), (Sphere(), (3, 2)))
    assert pman.n_elements == (10 + 3 * 2)
    assert pman.name == "(Sphere)x(Sphere)"
