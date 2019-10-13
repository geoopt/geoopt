import geoopt
import torch
import numpy as np


def test_scale_poincare():
    ball = geoopt.PoincareBallExact()
    sball = geoopt.Scaled(ball, 2)
    v = torch.arange(10).float() / 10
    np.testing.assert_allclose(
        ball.dist0(ball.expmap0(v)).item(),
        sball.dist0(sball.expmap0(v)).item(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sball.dist0(sball.expmap0(v)).item(),
        sball.norm(torch.zeros_like(v), v),
        atol=1e-5,
    )


def test_scale_poincare_learnable():
    ball = geoopt.PoincareBallExact()
    sball = geoopt.Scaled(ball, 2, learnable=True)
    v = torch.arange(10).float() / 10
    np.testing.assert_allclose(
        ball.dist0(ball.expmap0(v)).item(),
        sball.dist0(sball.expmap0(v)).item(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sball.dist0(sball.expmap0(v)).item(),
        sball.norm(torch.zeros_like(v), v),
        atol=1e-5,
    )


def test_scaling_compensates():
    ball = geoopt.PoincareBallExact()
    sball = geoopt.Scaled(ball, 2)
    rsball = geoopt.Scaled(sball, 0.5)
    v = torch.arange(10).float() / 10
    np.testing.assert_allclose(ball.expmap0(v), rsball.expmap0(v))


def test_scaling_getattr():
    ball = geoopt.PoincareBallExact()
    sball = geoopt.Scaled(ball, 2)
    pa, pb = sball.random(2, 10)
    # this one is representative and not present in __scaling__
    sball.geodesic(0.5, pa, pb)
