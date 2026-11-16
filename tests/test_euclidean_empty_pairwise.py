import pytest
import torch

import geoopt


@pytest.mark.parametrize("method", ["pairwise_inner", "cdist"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("manifold_shape", [(2, 3), (2, 2, 3)])
@pytest.mark.parametrize(
    "left_batch,right_batch,left_count,right_count",
    [
        ((), (), 0, 4),
        ((), (), 3, 0),
        ((), (), 0, 0),
        ((2, 1), (1, 3), 0, 4),
        ((0,), (1,), 3, 4),
        ((2, 1), (1, 3), 3, 4),
    ],
    ids=["empty-left", "empty-right", "empty-both", "broadcast-empty", "empty-batch", "nonempty"],
)
def test_higher_dimensional_pairwise_sets(
    method, dtype, manifold_shape, left_batch, right_batch, left_count, right_count
):
    generator = torch.Generator().manual_seed(864)
    left = torch.randn(
        (*left_batch, left_count, *manifold_shape), dtype=dtype, generator=generator
    ).transpose(-1, -2).requires_grad_()
    right = (
        torch.randn(
            (*right_batch, right_count, *manifold_shape), dtype=dtype, generator=generator
        ) + 3
    ).transpose(-1, -2).requires_grad_()
    reference_left = left.detach().clone().requires_grad_()
    reference_right = right.detach().clone().requires_grad_()
    ndim = len(manifold_shape)
    left_pairs = reference_left.unsqueeze(-ndim - 1)
    right_pairs = reference_right.unsqueeze(-ndim - 2)
    axes = tuple(range(-ndim, 0))
    if method == "pairwise_inner":
        expected = (left_pairs * right_pairs).sum(dim=axes)
    else:
        expected = torch.linalg.vector_norm(left_pairs - right_pairs, dim=axes)

    actual = getattr(geoopt.Euclidean(ndim=ndim), method)(left, right)

    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(left.grad, reference_left.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(right.grad, reference_right.grad, atol=tolerance, rtol=tolerance)
