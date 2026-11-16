import copy

import geoopt
import pytest
import torch


def optimizer(kind, point, maximize=False):
    if kind == "adam":
        return geoopt.optim.SparseRiemannianAdam([point], lr=0.02, maximize=maximize)
    if kind == "amsgrad":
        return geoopt.optim.SparseRiemannianAdam(
            [point], lr=0.02, amsgrad=True, maximize=maximize
        )
    options = {"lr": 0.02, "maximize": maximize}
    if kind != "sgd":
        options["momentum"] = 0.7
        options["dampening"] = 0.2 if kind == "momentum" else 0.0
        options["nesterov"] = kind == "nesterov"
    return geoopt.optim.SparseRiemannianSGD([point], **options)


def gradient_layouts(rows, values, shape):
    hybrid = torch.sparse_coo_tensor(rows.unsqueeze(0), values, shape)
    columns = torch.arange(shape[1]).repeat(rows.numel())
    coordinates = torch.stack((rows.repeat_interleave(shape[1]), columns))
    coordinate = torch.sparse_coo_tensor(coordinates, values.reshape(-1), shape)
    return hybrid, coordinate


@pytest.mark.parametrize("kind", ["sgd", "momentum", "nesterov", "adam", "amsgrad"])
@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_sparse_optimizer_layout_equivalence(kind, curved, coalesced, maximize):
    torch.manual_seed(1850)
    manifold = geoopt.Sphere() if curved else geoopt.Euclidean()
    initial = manifold.projx(torch.randn(31, 7, dtype=torch.float64))
    hybrid_point = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    coordinate_point = geoopt.ManifoldParameter(initial.clone(), manifold=manifold)
    hybrid_optimizer = optimizer(kind, hybrid_point, maximize)
    coordinate_optimizer = optimizer(kind, coordinate_point, maximize)
    for step in range(4):
        rows = torch.tensor([11, 2, 11, 5, 3, 2])
        values = torch.randn(rows.numel(), 7, dtype=torch.float64)
        if step == 2:
            values.zero_()
        hybrid_grad, coordinate_grad = gradient_layouts(rows, values, initial.shape)
        if coalesced:
            hybrid_grad, coordinate_grad = (
                hybrid_grad.coalesce(),
                coordinate_grad.coalesce(),
            )
        hybrid_point.grad = hybrid_grad
        coordinate_point.grad = coordinate_grad
        original_gradient = hybrid_grad.coalesce().values().clone()
        hybrid_optimizer.step()
        coordinate_optimizer.step()
        torch.testing.assert_close(hybrid_point, coordinate_point)
        torch.testing.assert_close(hybrid_grad.coalesce().values(), original_gradient)
        hybrid_state, coordinate_state = (
            hybrid_optimizer.state[hybrid_point],
            coordinate_optimizer.state[coordinate_point],
        )
        assert hybrid_state.keys() == coordinate_state.keys()
        for key in hybrid_state:
            torch.testing.assert_close(hybrid_state[key], coordinate_state[key])
    untouched = torch.tensor([0, 1, 4, 6, 30])
    torch.testing.assert_close(hybrid_point[untouched], initial[untouched])


@pytest.mark.parametrize("kind", ["sgd", "adam"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_embedding_training_matches_torch(kind, dtype):
    torch.manual_seed(1851)
    embedding = torch.nn.Embedding(100, 5, sparse=True).to(dtype)
    reference = copy.deepcopy(embedding)
    options = {"lr": 0.02}
    if kind == "adam":
        options["eps"] = 1e-30
    candidate_class = (
        geoopt.optim.SparseRiemannianSGD
        if kind == "sgd"
        else geoopt.optim.SparseRiemannianAdam
    )
    candidate_optimizer = candidate_class(embedding.parameters(), **options)
    reference_class = torch.optim.SGD if kind == "sgd" else torch.optim.SparseAdam
    reference_optimizer = reference_class(reference.parameters(), **options)
    for indices in [
        torch.tensor([4, 1, 4, 8]),
        torch.tensor([8, 4, 3]),
        torch.tensor([1, 8, 1]),
    ]:
        candidate_optimizer.zero_grad()
        reference_optimizer.zero_grad()
        embedding(indices).square().sum().backward()
        reference(indices).square().sum().backward()
        candidate_optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(embedding.weight, reference.weight)
