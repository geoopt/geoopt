import pytest
import torch

import geoopt


@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("transport", [False, True])
def test_euclidean_stiefel_qr_retraction_has_correct_gradients(batch_shape, transport):
    manifold = geoopt.Stiefel(canonical=False)
    generator = torch.Generator().manual_seed(42)
    point = manifold.random(*batch_shape, 4, 2, dtype=torch.float64, generator=generator)
    coordinates = torch.randn(
        *batch_shape, 4, 2, dtype=torch.float64, generator=generator
    ).requires_grad_()
    vector = manifold.proju(
        point, torch.randn(*batch_shape, 4, 2, dtype=torch.float64, generator=generator)
    )

    def retract(coordinates):
        tangent = 0.1 * manifold.proju(point, coordinates)
        if transport:
            return manifold.retr_transp(point, tangent, vector)
        return manifold.retr(point, tangent)

    assert torch.autograd.gradcheck(retract, (coordinates,))
