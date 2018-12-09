import torch.jit


def _compat_trace(fn, args):
    if tuple(map(int, torch.__version__.split(".")[:2])) < (1, 0):
        # torch.jit here does not support inplace ops
        return fn
    else:
        return torch.jit.trace(fn, args)


def create_traced_update(step, manifold, point, *buffers, **kwargs):
    """
    Make data dependent update for a given manifold and kwargs with given example point

    Parameters
    ----------
    step : callable
        step function to optimize
    manifold : Manifold
        manifold to build function for
    point : tensor
        example input point on manifold
    buffers : tensors
        optimizer dependent buffers
    kwargs : optional parameters for manifold

    Returns
    -------
    traced `step(point, grad, lr *buffers)` function
    """
    point = point.clone()
    grad = point.new(point.shape).normal_()
    lr = torch.tensor(0.001).type_as(grad)

    def partial(*args):
        step(manifold, *args, **kwargs)
        return args

    return _compat_trace(partial, (point, grad, lr) + buffers)
