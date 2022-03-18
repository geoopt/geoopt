from geoopt import StereographicProductManifold
from geoopt import PoincareBall, SphereProjection, Stereographic
import torch


def test_mobius_add_sub_coadd_cosub():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(6)
    y = pman.random(6)
    for kind in ["mobius_add", "mobius_coadd", "mobius_sub", "mobius_cosub"]:
        z = getattr(pman, kind)(x, y)
        assert pman._check_point_on_manifold(z)[0]

        for i, (manifold, _) in enumerate(manifolds):
            x_ = pman.take_submanifold_value(x, i)
            y_ = pman.take_submanifold_value(y, i)
            z_ = getattr(manifold, kind)(x_, y_)

            assert torch.isclose(pman.take_submanifold_value(z, i), z_).all()


def test_mobius_scalar_mul():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    r = torch.randn(4, 1)

    res = pman.mobius_scalar_mul(r, x)
    ress = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)

        ress.append(manifold.mobius_scalar_mul(r, x_))
    ress = torch.cat(ress, dim=-1)
    assert torch.isclose(ress, res).all()


def test_mobius_pointwise_mul():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    w1 = torch.randn(6)
    w2 = torch.randn(4, 6)

    res1 = pman.mobius_pointwise_mul(w1, x)
    res2 = pman.mobius_pointwise_mul(w2, x)
    ress1 = []
    ress2 = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        w1_ = pman.take_submanifold_value(w1, i)
        w2_ = pman.take_submanifold_value(w2, i)

        ress1.append(manifold.mobius_pointwise_mul(w1_, x_))
        ress2.append(manifold.mobius_pointwise_mul(w2_, x_))
    ress1 = torch.cat(ress1, dim=-1)
    assert torch.isclose(ress1, res1).all()
    ress2 = torch.cat(ress2, dim=-1)
    assert torch.isclose(ress2, res2).all()


def test_mobius_matvec():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    m1 = torch.randn(6, 6)
    m2 = torch.randn(4, 6, 6)

    res1 = pman.mobius_matvec(m1, x)
    res2 = pman.mobius_matvec(m2, x)
    ress1 = []
    ress2 = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        m1_ = pman.take_submanifold_matrix(m1, i)
        m2_ = pman.take_submanifold_matrix(m2, i)

        ress1.append(manifold.mobius_matvec(m1_, x_))
        ress2.append(manifold.mobius_matvec(m2_, x_))
    ress1 = torch.cat(ress1, dim=-1)
    assert torch.isclose(ress1, res1).all()
    ress2 = torch.cat(ress2, dim=-1)
    assert torch.isclose(ress2, res2).all()


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


def test_geodesic():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    y = pman.random(4, 6)
    t = torch.randn(4, 1)

    geodesic = pman.geodesic(t, x, y)
    geodesics = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        y_ = pman.take_submanifold_value(y, i)

        geodesics.append(manifold.geodesic(t, x_, y_))
    geodesics = torch.cat(geodesics, dim=-1)
    assert torch.isclose(geodesics, geodesic).all()


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


def test_dist0():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)

    res = pman.dist0(x)
    ress = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)

        ress.append(manifold.dist0(x_) ** 2)
    ress = sum(ress) ** 0.5
    assert torch.isclose(ress, res).all()


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


def test_logmap0():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(6)
    u = pman.logmap0(x)

    for i, (manifold, _) in enumerate(manifolds):
        u_ = manifold.logmap0(pman.take_submanifold_value(x, i))

        assert torch.isclose(pman.take_submanifold_value(u, i), u_).all()


def test_transp0():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    y = pman.random(4, 6)
    u = torch.randn(4, 6)
    x = pman.transp0(y, u)

    res = []
    for i, (manifold, _) in enumerate(manifolds):
        y_ = pman.take_submanifold_value(y, i)
        u_ = pman.take_submanifold_value(u, i)

        res.append(manifold.transp0(y_, u_))
    res = torch.cat(res, axis=-1)

    assert torch.isclose(res, x).all()


def test_transp0back():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    u = torch.randn(4, 6)
    y = pman.transp0back(x, u)

    res = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        u_ = pman.take_submanifold_value(u, i)

        res.append(manifold.transp0back(x_, u_))
    res = torch.cat(res, axis=-1)

    assert torch.isclose(res, y).all()


def test_gyration():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    y = pman.random(4, 6)
    z = pman.random(4, 6)
    r = pman.gyration(x, y, z)

    res = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)
        y_ = pman.take_submanifold_value(y, i)
        z_ = pman.take_submanifold_value(z, i)

        res.append(manifold.gyration(x_, y_, z_))
    res = torch.cat(res, axis=-1)

    assert torch.isclose(res, r).all()


def test_antipode():
    manifolds = (
        (PoincareBall(), 2),
        (SphereProjection(), 2),
        (Stereographic(), 2),
    )
    pman = StereographicProductManifold(*manifolds)
    x = pman.random(4, 6)
    a = pman.antipode(x)

    res = []
    for i, (manifold, _) in enumerate(manifolds):
        x_ = pman.take_submanifold_value(x, i)

        res.append(manifold.antipode(x_))
    res = torch.cat(res, axis=-1)

    assert torch.isclose(res, a).all()
