import geoopt.datasets
import geoopt
import pytest


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("depth", [3, 4])
@pytest.mark.parametrize("numberOfChildren", [2, 3])
@pytest.mark.parametrize("numberOfSiblings", [10, 30])
@pytest.mark.parametrize("c", [-1, 1])
def test_stereographic_data(dim, depth, numberOfChildren, numberOfSiblings, c):
    dataset = geoopt.datasets.stereographic.StereographicTreeDataset(
        dim=dim,
        ball=geoopt.Stereographic(c),
        depth=depth,
        numberOfChildren=numberOfChildren,
        numberOfsiblings=numberOfSiblings,
    )
    for point, labels, label in dataset:
        assert point.shape == (dim, )
        assert labels.shape == (depth + 1, )
        assert label == labels.max()