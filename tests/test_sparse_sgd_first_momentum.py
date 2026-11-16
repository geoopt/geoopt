import copy

import pytest
import torch

import geoopt


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "options",
    [
        {"momentum": 0.0},
        {"momentum": 0.9},
        {"momentum": 0.9, "dampening": 0.3},
        {"momentum": 0.9, "nesterov": True},
        {"momentum": 0.9, "maximize": True},
    ],
)
def test_sparse_embedding_momentum_matches_native_sgd(dtype, options):
    values = torch.tensor([[0.2, -0.4], [0.7, 0.1], [-0.5, 0.8]], dtype=dtype)
    actual = torch.nn.Parameter(values.clone())
    expected = torch.nn.Parameter(values.clone())
    optimizer = geoopt.optim.SparseRiemannianSGD([actual], lr=0.1, **options)
    reference = torch.optim.SGD([expected], lr=0.1, **options)
    indices = torch.tensor([2, 0, 2, 1])
    coefficients = values.new_tensor([[1.0, -2.0], [0.5, 0.7], [0.3, 0.6], [-0.2, 1.0]])

    for step in range(4):
        optimizer.zero_grad()
        reference.zero_grad()
        for point in (actual, expected):
            selected = torch.nn.functional.embedding(indices, point, sparse=True)
            (selected * coefficients * (step + 1)).sum().backward()
        assert actual.grad.is_sparse
        original_gradient = actual.grad.to_dense().clone()
        optimizer.step()
        reference.step()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.grad.to_dense(), original_gradient)
        if options["momentum"]:
            torch.testing.assert_close(
                optimizer.state[actual]["momentum_buffer"],
                reference.state[expected]["momentum_buffer"].to_dense(),
            )


@pytest.mark.parametrize("nesterov", [False, True])
def test_sparse_sphere_first_momentum_uses_only_tangent_gradient(nesterov):
    manifold = geoopt.Sphere()
    initial = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    point = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    gradient = initial.new_tensor([[2.0, 3.0, -1.0]])
    point.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), gradient, point.shape)
    optimizer = geoopt.optim.SparseRiemannianSGD([point], lr=0.1, momentum=0.9, nesterov=nesterov)
    tangent = initial.new_tensor([0.0, 3.0, -1.0])
    direction = tangent * (1.9 if nesterov else 1.0)
    expected_row = initial[0] - 0.1 * direction
    expected_row = expected_row / torch.linalg.vector_norm(expected_row)
    expected_momentum = tangent - (tangent * expected_row).sum() * expected_row

    optimizer.step()

    torch.testing.assert_close(point[0], expected_row)
    torch.testing.assert_close(point[1], initial[1], rtol=0, atol=0)
    buffer = optimizer.state[point]["momentum_buffer"]
    torch.testing.assert_close(buffer[0], expected_momentum)
    torch.testing.assert_close(buffer[1], torch.zeros_like(buffer[1]), rtol=0, atol=0)


def test_sparse_momentum_initializes_when_a_parameter_first_receives_a_gradient():
    early = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    delayed = torch.nn.Parameter(torch.zeros_like(early))
    optimizer = geoopt.optim.SparseRiemannianSGD(
        [early, delayed], lr=0.2, momentum=0.8, dampening=0.25
    )
    gradient = torch.sparse_coo_tensor(
        torch.tensor([[1]]), torch.tensor([[2.0, -3.0]], dtype=torch.float64), early.shape
    )
    early.grad = gradient.clone()
    optimizer.step()
    early.grad = None
    delayed.grad = gradient.clone()

    optimizer.step()

    torch.testing.assert_close(delayed, -0.2 * gradient.to_dense())
    torch.testing.assert_close(optimizer.state[delayed]["momentum_buffer"], gradient.to_dense())


def test_sparse_lazy_momentum_state_survives_checkpoint_and_inactive_rows():
    point = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.float64))
    optimizer = geoopt.optim.SparseRiemannianSGD([point], lr=0.1, momentum=0.7)
    point.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]),
        point.new_tensor([[1.0, -2.0], [0.5, 0.25]]),
        point.shape,
    )
    optimizer.step()
    resumed = torch.nn.Parameter(point.detach().clone())
    resumed_optimizer = geoopt.optim.SparseRiemannianSGD([resumed], lr=0.1, momentum=0.7)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    inactive_point = point[0].detach().clone()
    inactive_momentum = optimizer.state[point]["momentum_buffer"][0].clone()

    for parameter, opt in ((point, optimizer), (resumed, resumed_optimizer)):
        parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[2]]), parameter.new_tensor([[-0.25, 1.0]]), parameter.shape
        )
        opt.step()

    torch.testing.assert_close(resumed, point, rtol=0, atol=0)
    torch.testing.assert_close(point[0], inactive_point, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state[point]["momentum_buffer"][0], inactive_momentum, rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed_optimizer.state[resumed]["momentum_buffer"],
        optimizer.state[point]["momentum_buffer"],
        rtol=0,
        atol=0,
    )


def test_sparse_late_row_keeps_existing_lazy_recurrence():
    point = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    optimizer = geoopt.optim.SparseRiemannianSGD([point], lr=0.1, momentum=0.9, dampening=0.3)
    for row in (0, 2):
        point.grad = torch.sparse_coo_tensor(
            torch.tensor([[row]]), point.new_tensor([[2.0, -1.0]]), point.shape
        )
        optimizer.step()

    expected_buffer = point.new_tensor([1.4, -0.7])
    torch.testing.assert_close(optimizer.state[point]["momentum_buffer"][2], expected_buffer)
    torch.testing.assert_close(point[2], -0.1 * expected_buffer)
