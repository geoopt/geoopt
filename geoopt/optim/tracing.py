import torch.jit


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

    return torch.jit.trace(partial, (point, grad, lr) + buffers)
