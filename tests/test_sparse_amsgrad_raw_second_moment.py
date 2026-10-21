import copy
import io
import math

import geoopt
import pytest
import torch


DTYPES = [torch.float32, torch.float64]
INITIAL = [[1.0, -2.0], [0.5, 0.75], [-1.5, 2.25], [0.0, -0.5]]
HYPER = dict(lr=0.07, betas=(0.8, 0.9), eps=0.003)


class ScalarReference:
    """Independent lazy scalar recurrence, including the global step counter."""

    def __init__(self, amsgrad):
        self.point = copy.deepcopy(INITIAL)
        self.first = [[0.0, 0.0] for _ in INITIAL]
        self.second = [[0.0, 0.0] for _ in INITIAL]
        self.maximum = [[0.0, 0.0] for _ in INITIAL]
        self.step = 0
        self.amsgrad = amsgrad

    def advance(self, row_gradients):
        self.step += 1
        beta1, beta2 = HYPER["betas"]
        for row, gradient in row_gradients.items():
            for col, value in enumerate(gradient):
                self.first[row][col] = beta1 * self.first[row][col] + (1 - beta1) * value
                self.second[row][col] = beta2 * self.second[row][col] + (1 - beta2) * value**2
                self.maximum[row][col] = max(self.maximum[row][col], self.second[row][col])
                variance = self.maximum[row][col] if self.amsgrad else self.second[row][col]
                denominator = math.sqrt(variance / (1 - beta2**self.step)) + HYPER["eps"]
                self.point[row][col] -= HYPER["lr"] * self.first[row][col] / (
                    (1 - beta1**self.step) * denominator
                )


def make_embedding(dtype, amsgrad=True, manifold=False):
    embedding = torch.nn.Embedding(4, 2, sparse=True, dtype=dtype)
    if manifold:
        embedding.weight = geoopt.ManifoldParameter(
            torch.tensor(INITIAL, dtype=dtype), manifold=geoopt.Euclidean(ndim=1)
        )
    else:
        with torch.no_grad():
            embedding.weight.copy_(torch.tensor(INITIAL, dtype=dtype))
    optimizer = geoopt.optim.SparseRiemannianAdam([embedding.weight], amsgrad=amsgrad, **HYPER)
    return embedding, optimizer


def sparse_backward(embedding, optimizer, indices, coefficients):
    optimizer.zero_grad()
    values = embedding(torch.tensor(indices))
    weights = torch.tensor(coefficients, dtype=embedding.weight.dtype)
    (values * weights).sum().backward()
    assert embedding.weight.grad.is_sparse
    aggregated = {}
    for row, coefficient in zip(indices, coefficients):
        total = aggregated.setdefault(row, [0.0, 0.0])
        for col, value in enumerate(coefficient):
            total[col] += value
    optimizer.step()
    return aggregated


def assert_reference(embedding, optimizer, reference):
    dtype = embedding.weight.dtype
    rtol, atol = (3e-6, 3e-7) if dtype == torch.float32 else (1e-12, 1e-13)
    state = optimizer.state[embedding.weight]
    assert state["step"] == reference.step
    for actual, expected in [
        (embedding.weight, reference.point),
        (state["exp_avg"], reference.first),
        (state["exp_avg_sq"], reference.second),
    ]:
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype), rtol=rtol, atol=atol)
    if reference.amsgrad:
        torch.testing.assert_close(
            state["max_exp_avg_sq"], torch.tensor(reference.maximum, dtype=dtype), rtol=rtol, atol=atol
        )
    else:
        assert "max_exp_avg_sq" not in state


@pytest.mark.parametrize("dtype", DTYPES)
def test_constant_gradient_keeps_maximum_in_raw_second_moment_units(dtype):
    embedding, optimizer = make_embedding(dtype)
    reference = ScalarReference(amsgrad=True)
    for _ in range(6):
        gradients = sparse_backward(embedding, optimizer, [1], [[0.5, -0.25]])
        reference.advance(gradients)
    # Completing the recurrence before asserting also reproduces the parameter
    # error on upstream, rather than stopping at its first corrupted maximum.
    assert_reference(embedding, optimizer, reference)


@pytest.mark.parametrize("dtype", DTYPES)
def test_decreasing_gradient_retains_historical_uncorrected_maximum(dtype):
    embedding, optimizer = make_embedding(dtype)
    reference = ScalarReference(amsgrad=True)
    for coefficient in [[3.0, -2.0], [0.1, 0.05], [-0.2, 0.0], [0.0, -0.1], [0.3, 0.2]]:
        gradients = sparse_backward(embedding, optimizer, [2], [coefficient])
        reference.advance(gradients)
        assert_reference(embedding, optimizer, reference)


@pytest.mark.parametrize("dtype", DTYPES)
def test_embedding_duplicates_and_lazy_untouched_rows_follow_scalar_recurrence(dtype):
    embedding, optimizer = make_embedding(dtype)
    reference = ScalarReference(amsgrad=True)
    sequence = [
        ([0, 0, 2], [[0.5, 0.25], [0.25, -0.5], [1.0, -0.25]]),
        ([2, 3], [[-0.5, 0.25], [0.125, 0.75]]),
        ([0, 0, 3], [[-0.25, 0.5], [0.5, -0.25], [0.25, -0.125]]),
        ([2], [[0.125, -0.5]]),
    ]
    for indices, coefficients in sequence:
        previous_point = embedding.weight.detach().clone()
        previous_state = {
            key: value.clone() for key, value in optimizer.state[embedding.weight].items()
            if torch.is_tensor(value)
        }
        gradients = sparse_backward(embedding, optimizer, indices, coefficients)
        reference.advance(gradients)
        assert_reference(embedding, optimizer, reference)
        untouched = [row for row in range(4) if row not in gradients]
        torch.testing.assert_close(embedding.weight[untouched], previous_point[untouched], rtol=0, atol=0)
        for key, value in previous_state.items():
            torch.testing.assert_close(optimizer.state[embedding.weight][key][untouched], value[untouched],
                                       rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_weights_only_checkpoint_restores_raw_maximum_and_next_sparse_update(dtype):
    embedding, optimizer = make_embedding(dtype)
    reference = ScalarReference(amsgrad=True)
    for indices, coefficients in [
        ([0, 2], [[1.0, -0.5], [-0.25, 2.0]]),
        ([2], [[0.125, -0.25]]),
        ([0, 3], [[-0.5, 0.25], [0.25, 0.5]]),
    ]:
        reference.advance(sparse_backward(embedding, optimizer, indices, coefficients))
    assert_reference(embedding, optimizer, reference)
    checkpoint = io.BytesIO()
    torch.save({"embedding": embedding.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=True)
    resumed_embedding, resumed_optimizer = make_embedding(dtype)
    resumed_embedding.load_state_dict(restored["embedding"])
    resumed_optimizer.load_state_dict(restored["optimizer"])
    assert_reference(resumed_embedding, resumed_optimizer, reference)
    indices, coefficients = [0, 2, 2], [[0.25, -0.5], [-0.25, 0.5], [0.125, 0.25]]
    gradients = sparse_backward(embedding, optimizer, indices, coefficients)
    resumed_gradients = sparse_backward(resumed_embedding, resumed_optimizer, indices, coefficients)
    assert gradients == resumed_gradients
    reference.advance(gradients)
    assert_reference(embedding, optimizer, reference)
    assert_reference(resumed_embedding, resumed_optimizer, reference)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ordinary_sparse_adam_retains_its_existing_lazy_recurrence(dtype):
    embedding, optimizer = make_embedding(dtype, amsgrad=False)
    reference = ScalarReference(amsgrad=False)
    for indices, coefficients in [
        ([0, 2], [[0.5, -0.25], [1.0, -0.5]]),
        ([2], [[0.125, 0.25]]),
        ([0, 0], [[-0.25, 0.5], [0.125, 0.25]]),
    ]:
        reference.advance(sparse_backward(embedding, optimizer, indices, coefficients))
        assert_reference(embedding, optimizer, reference)


@pytest.mark.parametrize("dtype", DTYPES)
def test_manifold_parameter_sparse_amsgrad_preserves_raw_state(dtype):
    embedding, optimizer = make_embedding(dtype, manifold=True)
    reference = ScalarReference(amsgrad=True)
    for indices, coefficients in [
        ([1], [[0.5, -0.25]]),
        ([1, 2], [[0.125, -0.5], [-0.5, 0.25]]),
        ([2], [[0.25, 0.125]]),
    ]:
        reference.advance(sparse_backward(embedding, optimizer, indices, coefficients))
        assert_reference(embedding, optimizer, reference)
