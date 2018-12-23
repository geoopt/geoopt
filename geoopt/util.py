import torch
import itertools

__all__ = ["svd"]


def svd(x):
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    batches = x.shape[:-2]
    if batches:
        n, m = x.shape[-2:]
        k = min(n, m)
        U, d, V = x.new(*batches, n, k), x.new(*batches, k), x.new(*batches, m, k)
        for idx in itertools.product(*map(range, batches)):
            U[idx], d[idx], V[idx] = torch.svd(x[idx])
        return U, d, V
    else:
        return torch.svd(x)


def broadcast_function_apply(input, output, shape, fun):
    if not isinstance(input, (list, tuple)):
        input = [input]
    if isinstance(output, (list, tuple)):
        if shape:
            for idx in itertools.product(*map(range, shape)):
                small_output = fun(*(inp[idx] for inp in input))
                for out, storage in zip(small_output, output):
                    storage[idx] = out
        else:
            output = output.__class__(fun(*input))
    else:
        if shape:
            for idx in itertools.product(*map(range, shape)):
                small_output = fun(*(inp[idx] for inp in input))
                output[idx] = small_output
        else:
            output = fun(*input)
    return output


def sym(x):
    return 0.5 * (x.transpose(-1, -2) + x)
