import torch
import itertools

__all__ = ["svd", "qr", "sym", "extract_diag"]


def svd(x):
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    batches = x.shape[:-2]
    if batches:
        # in most cases we do not require gradients when applying svd (e.g. in projection)
        assert not x.requires_grad
        n, m = x.shape[-2:]
        k = min(n, m)
        U, d, V = x.new(*batches, n, k), x.new(*batches, k), x.new(*batches, m, k)
        for idx in itertools.product(*map(range, batches)):
            U[idx], d[idx], V[idx] = torch.svd(x[idx])
        return U, d, V
    else:
        return torch.svd(x)


def qr(x):
    # vectorized version as svd
    batches = x.shape[:-2]
    if batches:
        # in most cases we do not require gradients when applying qr (e.g. in retraction)
        assert not x.requires_grad
        n, m = x.shape[-2:]
        Q, R = x.new(*batches, n, m), x.new(*batches, m, m)
        for idx in itertools.product(*map(range, batches)):
            Q[idx], R[idx] = torch.qr(x[idx])
        return Q, R
    else:
        return torch.qr(x)


def sym(x):
    return 0.5 * (x.transpose(-1, -2) + x)


def extract_diag(x):
    n, m = x.shape[-2:]
    k = min(n, m)
    return x[..., torch.arange(k), torch.arange(k)]
