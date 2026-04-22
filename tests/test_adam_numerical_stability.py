import torch
import geoopt


def test_adam_stabilizes_off_manifold():
    """Test that RiemannianAdam handles points initially off-manifold."""
    torch.manual_seed(42)
    manifold = geoopt.PoincareBall()

    # Create weights initially outside the manifold (norm > 1)
    weight = torch.randn(10, 5) * 0.9
    weight = geoopt.ManifoldParameter(weight, manifold=manifold)

    # Verify weights are off-manifold initially
    on_manifold, _ = manifold._check_point_on_manifold(weight)
    assert not on_manifold, "Weights should start off-manifold"

    optimizer = geoopt.optim.RiemannianAdam([weight], lr=0.1)

    x = torch.randn(32, 5)

    # Run several optimization steps - should not produce NaN
    for step in range(10):
        optimizer.zero_grad()

        output = torch.matmul(x, weight.T)
        loss = output.mean()

        assert not torch.isnan(loss), f"Loss became NaN at step {step}"

        loss.backward()
        optimizer.step()

    # After optimization, weights should be on manifold
    on_manifold, _ = manifold._check_point_on_manifold(weight)
    assert on_manifold, "Weights should be on manifold after optimization"


def test_adam_near_boundary_stays_stable():
    """Test stability when approaching manifold boundary.

    Regression test for edge cases where points are very close to
    the manifold boundary.
    """
    torch.manual_seed(42)
    manifold = geoopt.PoincareBall()

    # Initialize close to but inside the boundary
    weight = torch.randn(10, 5) * 0.01
    weight = manifold.expmap0(weight)
    weight = geoopt.ManifoldParameter(weight, manifold=manifold)

    optimizer = geoopt.optim.RiemannianAdam([weight], lr=0.1)

    x = torch.randn(32, 5)

    for step in range(50):
        optimizer.zero_grad()

        output = torch.matmul(x, weight.T)
        loss = output.mean()

        assert not torch.isnan(loss), f"Loss became NaN at step {step}"

        loss.backward()
        optimizer.step()


def test_adam_with_stabilize_param():
    """Test that stabilize parameter works together with the numerical stability fix."""
    torch.manual_seed(42)
    manifold = geoopt.PoincareBall()

    # Start off-manifold
    weight = torch.randn(10, 5) * 0.9
    weight = geoopt.ManifoldParameter(weight, manifold=manifold)

    optimizer = geoopt.optim.RiemannianAdam([weight], lr=0.1, stabilize=5)

    assert optimizer.param_groups[0]["stabilize"] == 5

    x = torch.randn(32, 5)

    for step in range(20):
        optimizer.zero_grad()

        output = torch.matmul(x, weight.T)
        loss = output.mean()

        assert not torch.isnan(loss), f"Loss became NaN at step {step}"

        loss.backward()
        optimizer.step()


def test_adam_multiple_manifolds():
    """Test with different manifold types."""
    torch.manual_seed(42)

    manifolds = [
        geoopt.PoincareBall(),
        geoopt.SphereProjection(),
        geoopt.Lorentz(),
    ]

    for manifold in manifolds:
        weight = torch.randn(10, 5)
        if hasattr(manifold, "projx"):
            weight = manifold.projx(weight)
        weight = geoopt.ManifoldParameter(weight, manifold=manifold)

        optimizer = geoopt.optim.RiemannianAdam([weight], lr=0.01)

        x = torch.randn(32, 5)

        for step in range(10):
            optimizer.zero_grad()

            output = torch.matmul(x, weight.T)
            loss = output.mean()

            assert not torch.isnan(loss), f"Loss became NaN for manifold {manifold.name}"

            loss.backward()
            optimizer.step()
