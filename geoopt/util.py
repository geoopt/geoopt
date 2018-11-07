import torch
import itertools


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
