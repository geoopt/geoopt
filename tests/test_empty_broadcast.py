import pytest
import torch

import geoopt
from geoopt.utils import broadcast_shapes


@pytest.mark.parametrize(
    "shapes",
    [
        ((0,), (1,)),
        ((1,), (0,)),
        ((0, 3), ()),
        ((0, 3), (0, 3)),
        ((2, 0, 3), (1, 1, 3)),
        ((1, 0, 3), (5, 1, 1), (0, 3)),
        ((), ()),
        ((3, 1), (1, 4)),
        ((2, 3), (3,)),
        ((1, 1), ()),
    ],
)
def test_broadcast_shapes_agrees_with_torch(shapes):
    assert broadcast_shapes(*shapes) == tuple(torch.broadcast_shapes(*shapes))


@pytest.mark.parametrize("shapes", [((0,), (2,)), ((2, 3), (0, 3))])
def test_zero_dimension_is_incompatible_with_nonunit_dimension(shapes):
    with pytest.raises(RuntimeError):
        torch.broadcast_shapes(*shapes)
    with pytest.raises(ValueError):
        broadcast_shapes(*shapes)


@pytest.mark.parametrize("kind", ["euclidean", "sphere"])
@pytest.mark.parametrize("shape", [(0, 3), (2, 0, 3)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_empty_inner_product_has_no_loss_or_tangent_gradient(kind, shape, keepdim):
    manifold = geoopt.Euclidean(ndim=1) if kind == "euclidean" else geoopt.Sphere()
    x = torch.empty(shape, dtype=torch.float64)
    u = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)

    inner = manifold.inner(x, u, keepdim=keepdim)

    assert inner.shape == shape[:-1] + ((1,) if keepdim else ())
    loss = inner.sum()
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(u.grad, torch.zeros_like(u))


@pytest.mark.parametrize("operation", ["proju", "egrad2rgrad", "component_inner"])
def test_euclidean_broadcast_operations_preserve_empty_batch(operation):
    manifold = geoopt.Euclidean(ndim=1)
    x = torch.empty(2, 0, 3, dtype=torch.float64)
    u = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)

    result = getattr(manifold, operation)(x, u)

    assert result.shape == x.shape
    assert result.numel() == 0
    result.sum().backward()
    torch.testing.assert_close(u.grad, torch.zeros_like(u))
