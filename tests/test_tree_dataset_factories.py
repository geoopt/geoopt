import pytest
import torch

import geoopt
from geoopt.datasets.stereographic import StereographicTreeDataset


@pytest.mark.parametrize("global_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("manifold_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim,children", [(2, 2), (2, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("curvature", [-1.0, 1.0])
def test_tree_geometry_uses_the_manifold_dtype(
    global_dtype, manifold_dtype, dim, children, curvature
):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(global_dtype)
        ball = geoopt.Stereographic(torch.tensor(curvature, dtype=manifold_dtype))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(312)
            dataset = StereographicTreeDataset(
                ball, dim, depth=2, numberOfChildren=children,
                numberOfsiblings=2, dist_children=0.123456789012345,
                sigma_sibling=0.01,
            )
        assert dataset.root.dtype == manifold_dtype
        assert dataset.origin_data.dtype == manifold_dtype
        assert dataset.data.dtype == manifold_dtype
        directions = dataset.get_directions(dataset.origin_data[1], children)
        assert directions.dtype == manifold_dtype
        assert directions.device == ball.device
        assert torch.isfinite(dataset.origin_data).all()
        assert torch.isfinite(dataset.data).all()
        assert dataset.origin_labels.dtype == torch.long
        assert dataset.labels.dtype == torch.long
        assert len(dataset) == 2 * (1 + children + children**2)
        distances = ball.dist(dataset.root, dataset.origin_data[1:children + 1])
        tolerance = 1e-6 if manifold_dtype == torch.float32 else 1e-12
        torch.testing.assert_close(
            distances, torch.full((children,), dataset.dist_children, dtype=manifold_dtype),
            atol=tolerance, rtol=tolerance,
        )
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dim,children", [(2, 2), (3, 3)])
def test_tree_geometry_follows_a_cuda_manifold(dim, children):
    ball = geoopt.PoincareBall().to(device="cuda", dtype=torch.float64)
    dataset = StereographicTreeDataset(
        ball, dim, depth=1, numberOfChildren=children, dist_children=0.1,
    )
    assert dataset.root.device == ball.device
    assert dataset.origin_data.device == ball.device
    assert dataset.data.device == ball.device
    assert torch.isfinite(dataset.data).all()
