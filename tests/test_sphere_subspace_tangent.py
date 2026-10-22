import pytest
import torch

import geoopt


@pytest.mark.parametrize("constraint", ["intersection", "complement"])
@pytest.mark.parametrize("batch", [False, True])
def test_constrained_sphere_rejects_normal_to_subspace(constraint, batch):
    identity = torch.eye(4, dtype=torch.float64)
    kwargs = {constraint: identity[:, :3] if constraint == "intersection" else identity[:, 3:]}
    manifold = geoopt.Sphere(**kwargs)
    x = identity[0]
    u = identity[3]  # orthogonal to x, but outside the allowed subspace
    if batch:
        x = x.expand(2, 4)
        u = torch.stack((identity[1], u))
    assert not manifold.check_vector_on_tangent(x, u)
    with pytest.raises(ValueError, match="subspace"):
        manifold.assert_check_vector_on_tangent(x, u)
    projected = manifold.proju(x, u)
    assert manifold.check_vector_on_tangent(x, projected)


@pytest.mark.parametrize("constraint", ["intersection", "complement"])
def test_constrained_sphere_accepts_vectors_in_both_constraints(constraint):
    identity = torch.eye(4, dtype=torch.float64)
    kwargs = {constraint: identity[:, :3] if constraint == "intersection" else identity[:, 3:]}
    manifold = geoopt.Sphere(**kwargs)
    x = identity[0]
    assert manifold.check_vector_on_tangent(x, identity[1])
    assert manifold.check_vector_on_tangent(x, torch.zeros_like(x))
    assert not manifold.check_vector_on_tangent(x, x)


def test_unconstrained_sphere_tangent_check_is_unchanged():
    identity = torch.eye(4, dtype=torch.float64)
    assert geoopt.Sphere().check_vector_on_tangent(identity[0], identity[3])
