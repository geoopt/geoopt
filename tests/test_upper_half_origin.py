import pytest
import torch

import geoopt


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (2, 1, 3, 3), (0, 3, 3)])
@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize("dtype", [None, torch.float64, torch.complex64, torch.complex128])
def test_upper_half_origin_preserves_requested_shape(shape, as_tuple, dtype):
    manifold = geoopt.UpperHalf()
    args = (shape,) if as_tuple else shape

    point = manifold.origin(*args, dtype=dtype, device="cpu")

    expected_dtype = torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64
    assert point.shape == shape
    assert point.dtype == expected_dtype
    assert point.device == torch.device("cpu")
    torch.testing.assert_close(point.real, torch.zeros(shape, dtype=point.real.dtype))
    expected_imaginary = torch.zeros(shape, dtype=point.real.dtype)
    for i in range(shape[-1]):
        expected_imaginary[..., i, i] = 1
    torch.testing.assert_close(point.imag, expected_imaginary)
    manifold.assert_check_point_on_manifold(point)


def test_product_origin_with_upper_half_has_the_requested_batch():
    upper = geoopt.UpperHalf()
    manifold = geoopt.ProductManifold((upper, (3, 3)), (geoopt.BoundedDomain(), (2, 2)))

    point = manifold.origin(2, 1, manifold.n_elements, dtype=torch.complex128)

    assert point.shape == (2, 1, 13)
    upper_point, bounded_point = manifold.unpack_tensor(point)
    assert upper_point.shape == (2, 1, 3, 3)
    assert bounded_point.shape == (2, 1, 2, 2)
    manifold.assert_check_point_on_manifold(point)
