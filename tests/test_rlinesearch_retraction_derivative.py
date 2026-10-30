import pytest
import torch

import geoopt


def _line_search_problem(manifold, values, target):
    point = geoopt.ManifoldParameter(values.clone(), manifold=manifold)
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method="wolfe", cg_method="fr"
    )

    def objective(value):
        return 0.5 * (value - target).square().sum()

    def closure():
        optimizer.zero_grad()
        loss = objective(point)
        loss.backward()
        return loss.item()

    optimizer.closure = closure
    optimizer._init_loss()
    return point, optimizer, objective


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5])
def test_wolfe_derivative_matches_curve_finite_difference(scale, alpha):
    manifold = geoopt.CanonicalStiefel()
    if scale != 1:
        manifold = geoopt.Scaled(manifold, scale).double()
    x = torch.eye(4, 2, dtype=torch.float64)
    direction = torch.tensor(
        [[0.0, -0.2], [0.2, 0.0], [0.3, 0.1], [-0.1, 0.2]], dtype=x.dtype
    )
    target = torch.tensor(
        [[1.2, -0.3], [0.4, 0.8], [0.2, -0.5], [0.1, 0.7]], dtype=x.dtype
    )
    point, optimizer, objective = _line_search_problem(manifold, x, target)
    optimizer.state[point]["search_direction"] = direction

    epsilon = 1e-6
    expected = (
        objective(manifold.retr(x, (alpha + epsilon) * direction))
        - objective(manifold.retr(x, (alpha - epsilon) * direction))
    ).item() / (2 * epsilon)
    actual = optimizer._derphi(alpha)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)
    state = optimizer.state[point]
    transported = manifold.transp_follow_retr(x, alpha * direction, direction)
    # The derivative must not replace the isometric transport used by CG.
    torch.testing.assert_close(state["search_transported"], transported)
    torch.testing.assert_close(
        manifold.inner(state["new_point"], state["search_transported"]),
        manifold.inner(x, direction),
    )
    torch.testing.assert_close(point, x)


@pytest.mark.parametrize("geometry", ["euclidean", "sphere", "poincare", "scaled"])
def test_unsupported_retraction_jvp_preserves_transport_fallback(geometry):
    if geometry == "sphere":
        manifold = geoopt.Sphere()
        x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    else:
        manifold = (
            geoopt.PoincareBall().double()
            if geometry == "poincare"
            else geoopt.Euclidean(ndim=1)
        )
        if geometry == "scaled":
            manifold = geoopt.Scaled(manifold, 2).double()
        x = torch.tensor([0.2, 0.1, -0.1], dtype=torch.float64)
    target = torch.tensor([0.3, -0.4, 0.2], dtype=x.dtype)
    point, optimizer, _ = _line_search_problem(manifold, x, target)
    direction = optimizer.state[point]["search_direction"].clone()
    with pytest.raises(NotImplementedError):
        manifold.retr_jvp(x, 0.1 * direction, direction)

    actual = optimizer._derphi(0.1)

    state = optimizer.state[point]
    expected = manifold.inner(
        state["new_point"], state["new_grad"], state["search_transported"]
    ).sum().item()
    assert actual == expected
    torch.testing.assert_close(
        state["retraction_velocity"], state["search_transported"]
    )


def test_line_search_does_not_hide_retraction_jvp_errors():
    class BrokenDerivative(geoopt.Euclidean):
        def retr_jvp(self, x, u, v):
            raise ValueError("invalid derivative input")

    x = torch.tensor([0.2, 0.1], dtype=torch.float64)
    _, optimizer, _ = _line_search_problem(
        BrokenDerivative(ndim=1), x, torch.zeros_like(x)
    )

    with pytest.raises(ValueError, match="invalid derivative input"):
        optimizer._derphi(0.1)


def test_armijo_does_not_request_retraction_jvp():
    class UnusedDerivative(geoopt.Euclidean):
        def retr_jvp(self, x, u, v):
            raise AssertionError("Armijo does not need a retraction derivative")

    point = geoopt.ManifoldParameter(
        torch.tensor([0.2, 0.1], dtype=torch.float64),
        manifold=UnusedDerivative(ndim=1),
    )
    optimizer = geoopt.optim.RiemannianLineSearch(
        [point], line_search_method="armijo"
    )
    before = point.square().sum().item()

    def closure():
        optimizer.zero_grad()
        loss = point.square().sum()
        loss.backward()
        return loss.item()

    assert optimizer.step(closure) < before
