from geoopt import StereographicProductManifold
from geoopt import PoincareBall, SphereProjection, Stereographic
import torch


def test_expmap0():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    u = torch.randn(6)
    x = pman.expmap0(u)
    assert pman._check_point_on_manifold(x)[0]

    for i, (manifold, _) in enumerate(manifolds):
        x_ = manifold.expmap0(pman.take_submanifold_value(u, i))

        assert torch.isclose(pman.take_submanifold_value(x, i), x_).all()


def test_mobius_add():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(6)
    y = pman.random(6)
    z = pman.mobius_add(x, y)
    assert pman._check_point_on_manifold(z)[0]

    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        y_ = pman.take_submanifold_value(y, i)
        z_ = manifold.mobius_add(x_, y_)

        assert torch.isclose(pman.take_submanifold_value(z, i), z_).all()


def test_dist2plane():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(6)
    p = pman.random(6)
    a = torch.randn(6)

    dist = pman.dist2plane(x, p, a)
    dists = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        p_ = pman.take_submanifold_value(p, i)
        a_ = pman.take_submanifold_value(a, i)

        dists.append(manifold.dist2plane(x_, p_, a_))
    dists = torch.tensor(dists) ** 2
    assert torch.isclose(dists.sum().sqrt(), dist).all()


def test_geodesic_unit():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    u = torch.randn(4, 6)
    t = torch.randn(4, 1)

    geodesic = pman.geodesic_unit(t, x, u)
    geodesics = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        u_ = pman.take_submanifold_value(u, i)

        geodesics.append(manifold.geodesic_unit(t, x_, u_))
    geodesics = torch.cat(geodesics, dim=-1)
    assert torch.isclose(geodesics, geodesic).all()
