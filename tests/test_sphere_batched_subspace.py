import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("manifold_type", [geoopt.Sphere, geoopt.SphereExact])
@pytest.mark.parametrize("constraint", ["intersection", "complement"])
@pytest.mark.parametrize("batch_shape", [(), (2,), (4, 2), (4, 1)])
def test_batched_subspace_projection(manifold_type, constraint, batch_shape, dtype):
    generator = torch.Generator().manual_seed(217)
    subspaces = torch.randn(2, 5, 2, dtype=dtype, generator=generator)
    raw = torch.randn(*batch_shape, 5, dtype=dtype, generator=generator)
    manifold = manifold_type(**{constraint: subspaces})
    batch_raw = raw.expand(*torch.broadcast_shapes(batch_shape, (2,)), 5)
    expected = torch.stack(
        [
            manifold_type(**{constraint: subspaces[i]}).projx(batch_raw[..., i, :])
            for i in range(2)
        ],
        dim=-2,
    )
    actual = manifold.projx(raw)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
    tangent = manifold.proju(actual, batch_raw)
    assert tangent.shape == expected.shape
    assert manifold.check_point_on_manifold(actual)
    assert manifold.check_vector_on_tangent(actual, tangent)
