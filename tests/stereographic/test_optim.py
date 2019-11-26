import random
import geoopt
import torch
import numpy as np
import pytest
from geoopt.manifolds import Stereographic
from geoopt.manifolds import StereographicExact
from geoopt.manifolds import PoincareBall
from geoopt.manifolds import PoincareBallExact

@pytest.fixture("function", autouse=True, params=range(30, 40))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


# TODO: float32 just causes impossibly many problems!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@pytest.fixture("function", params=[torch.float64]) #, torch.float32])
def dtype(request):
    return request.param


@pytest.fixture
def c(seed, dtype):
    # test positive and negative curvatures
    if seed < 35:
        c = 3 * torch.rand(1, dtype=dtype)+0.0000001
    else:
        c = 3 * torch.rand(1, dtype=dtype)-0.0000001
    return c


@pytest.fixture("function", params=[1,2,3,4])
def manifold(request, c, dtype):
    if request.param == 1:
        return Stereographic(K=-c)
    elif request.param == 2:
        return StereographicExact(K=-c)
    elif request.param == 3:
        c=c.abs().to(dtype)
        torch.set_default_dtype(dtype)
        return PoincareBall(c=c)
    else:
        torch.set_default_dtype(dtype)
        c=c.abs().to(dtype)
        return PoincareBallExact(c=c)

@pytest.fixture
def R(c):
    R = 1.0/(c.abs().sqrt())
    return R

def create_random_point(c, R, manifold, dtype):

    # create point by initializing a random normal vector at the origin
    point = torch.empty(2, dtype=dtype)
    torch.nn.init.normal_(point, 0.0, 1.0)

    # normalize the vector
    point_norm = point.norm(p=2, dim=-1)
    point = point / point_norm

    # rescale the vector randomly
    if c > 0:
        # make sure target is within radius of Poincare ball
        point *= torch.rand(1, dtype=dtype) * ((1 - 0.000000001) * R)
        assert (point.norm(p=2, dim=-1) < R)
    else:
        # test targets anywhere (inside or outside of radius) for spherical
        # geometry
        point *= torch.rand(1, dtype=dtype) * (4 * R)

    # assign the point to the manifold
    point = geoopt.ManifoldParameter(point, manifold=manifold)

    # assert that the point is on the manifold
    manifold.assert_check_point_on_manifold(point)

    return point


@pytest.fixture
def start(manifold, c, R, seed, dtype):
    return create_random_point(c, R, manifold, dtype)

@pytest.fixture
def target(manifold, c, R, seed, dtype):
    point = create_random_point(c, R, manifold, dtype)
    point.requires_grad = False
    return point

@pytest.fixture("function", params=[0.1])
def lr(request):
    return request.param

def test_adam_stereographic(manifold, c, lr, start, target):

    # create optimizer closure
    def closure():
        optim.zero_grad()
        loss = manifold.dist2(start, target)
        loss.backward()
        return loss.item()

    optim = geoopt.optim.RiemannianAdam([start],
                                        lr=lr,
                                        stabilize=1)

    for _ in range(2000):
        _ = optim.step(closure)

    np.testing.assert_allclose(start.data, target.data, atol=1e-5, rtol=1e-5)