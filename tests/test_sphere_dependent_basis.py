import geoopt
import pytest
import torch


@pytest.mark.parametrize("constraint", ["intersection", "complement"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_dependent_sphere_basis_preserves_its_span(constraint, dtype, wide, exact):
    identity = torch.eye(6, dtype=dtype)
    basis = torch.stack((identity[:, 0], identity[:, 1], identity[:, 0] + identity[:, 1], identity[:, 0] * 0), dim=-1)
    if wide:
        basis = torch.cat((basis, basis), dim=-1)
    manifold_class = geoopt.SphereExact if exact else geoopt.Sphere
    manifold = manifold_class(**{constraint: basis})
    projector = torch.diag(torch.tensor([1, 1, 0, 0, 0, 0], dtype=dtype))
    if constraint == "complement":
        projector = identity - projector
    torch.testing.assert_close(manifold.projector, projector)
    torch.manual_seed(1950)
    sample = manifold.random_uniform(8, 6)
    torch.testing.assert_close(sample @ projector, sample)
    torch.testing.assert_close(sample.norm(dim=-1), torch.ones(8, dtype=dtype))
    tangent = manifold.proju(sample, torch.randn_like(sample))
    torch.testing.assert_close(tangent @ projector, tangent)
    torch.testing.assert_close((sample * tangent).sum(-1), torch.zeros(8, dtype=dtype), atol=1e-6, rtol=0)


@pytest.mark.parametrize("constraint", ["intersection", "complement"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sphere_basis_batches_can_have_different_ranks(constraint, dtype):
    identity = torch.eye(6, dtype=dtype)
    basis = torch.zeros(2, 6, 4, dtype=dtype)
    basis[0, :, :2] = identity[:, :2]
    basis[0, :, 2] = identity[:, 0]
    basis[1, :, :3] = identity[:, :3]
    basis[1, :, 3] = identity[:, 1]
    expected = torch.stack((torch.diag(torch.tensor([1, 1, 0, 0, 0, 0], dtype=dtype)), torch.diag(torch.tensor([1, 1, 1, 0, 0, 0], dtype=dtype))))
    if constraint == "complement":
        expected = identity - expected
    manifold = geoopt.Sphere(**{constraint: basis})
    torch.testing.assert_close(manifold.projector, expected)
    assert torch.equal(torch.linalg.matrix_rank(manifold.projector), torch.tensor([2, 3]) if constraint == "intersection" else torch.tensor([4, 3]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["empty_span", "line_span", "full_complement", "codimension_one"])
def test_sphere_rejects_subspaces_without_a_continuous_sphere(dtype, case):
    identity = torch.eye(5, dtype=dtype)
    if case == "empty_span":
        kwargs = {"intersection": torch.zeros(5, 3, dtype=dtype)}
    elif case == "line_span":
        kwargs = {"intersection": identity[:, :1].expand(5, 3)}
    elif case == "full_complement":
        kwargs = {"complement": identity}
    else:
        kwargs = {"complement": torch.cat((identity[:, :4], identity[:, :1]), dim=-1)}
    with pytest.raises(ValueError):
        geoopt.Sphere(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_complement_preserves_unconstrained_sphere(dtype):
    manifold = geoopt.Sphere(complement=torch.empty(6, 0, dtype=dtype))
    torch.testing.assert_close(manifold.projector, torch.eye(6, dtype=dtype))


@pytest.mark.parametrize("constraint", ["intersection", "complement"])
def test_full_rank_sphere_basis_keeps_projector_and_gradients(constraint):
    torch.manual_seed(1951)
    basis = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    q = torch.linalg.qr(basis).Q
    reference = q @ q.T
    if constraint == "complement":
        reference = torch.eye(6, dtype=torch.float64) - reference
    actual = geoopt.Sphere(**{constraint: basis}).projector
    torch.testing.assert_close(actual, reference)
    weight = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad((actual * weight).sum(), basis)[0]
    reference_gradient = torch.autograd.grad((reference * weight).sum(), basis)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)
    assert torch.autograd.gradcheck(lambda x: geoopt.Sphere(**{constraint: x}).projector, (basis,))
