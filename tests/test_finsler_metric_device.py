import pytest
import torch

from geoopt.manifolds.siegel.vvd_metrics import FinslerMinimumEntropyMetric


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3), (2, 4, 3)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_minimum_entropy_metric_preserves_values_and_gradients(dtype, shape, keepdim):
    metric = FinslerMinimumEntropyMetric(rank=3)
    values = torch.arange(1, 1 + shape[-1], dtype=dtype).expand(shape).clone()
    values.requires_grad_()
    original_weights = metric.weights.clone()
    result = metric.compute_metric(values, keepdim=keepdim)
    expected = 2 * values[..., 1] + 4 * values[..., 2]
    if keepdim:
        expected = expected.unsqueeze(-1)

    torch.testing.assert_close(result, expected)
    assert result.dtype == dtype
    result.sum().backward()
    torch.testing.assert_close(values.grad, values.new_tensor([0, 2, 4]).expand(shape))
    torch.testing.assert_close(metric.weights, original_weights, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3), (2, 4, 3)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_minimum_entropy_metric_follows_non_cpu_input_device(dtype, shape, keepdim):
    metric = FinslerMinimumEntropyMetric(rank=3)
    values = torch.empty(shape, dtype=dtype, device="meta")
    result = metric.compute_metric(values, keepdim=keepdim)

    assert result.device == values.device
    assert result.dtype == values.dtype
    expected_shape = shape[:-1] + ((1,) if keepdim else ())
    assert result.shape == expected_shape
    assert metric.weights.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_minimum_entropy_metric_cuda_values_and_gradients(dtype):
    metric = FinslerMinimumEntropyMetric(rank=3)
    values = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device="cuda", requires_grad=True)
    result = metric.compute_metric(values, keepdim=False)

    torch.testing.assert_close(result, values.new_tensor([16, 34]))
    result.sum().backward()
    torch.testing.assert_close(values.grad, values.new_tensor([[0, 2, 4], [0, 2, 4]]))
