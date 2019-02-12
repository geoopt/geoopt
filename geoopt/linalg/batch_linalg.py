import torch
from . import _expm

__all__ = ["svd", "qr", "sym", "extract_diag", "matrix_rank", "expm"]


@torch.jit.script
def svd(x):  # pragma: no cover
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        # 17 milliseconds on my mac to check that condition, that is low overhead
        result = torch.svd(x)
    else:
        batches = x.shape[:-2]
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        U, D, V = [], [], []
        # I wish I had a parallel_for
        for i in range(flat.shape[0]):
            u, d, v = torch.svd(slices[i])
            U += [u]
            D += [d]
            V += [v]
        U = torch.stack(U).view(batches + U[0].shape)
        D = torch.stack(D).view(batches + D[0].shape)
        V = torch.stack(V).view(batches + V[0].shape)
        result = U, D, V
    return result


@torch.jit.script
def qr(x):  # pragma: no cover
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        result = torch.qr(x)
    else:
        batches = x.shape[:-2]
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        Q, R = [], []
        # I wish I had a parallel_for
        for i in range(flat.shape[0]):
            q, r = torch.qr(slices[i])
            Q += [q]
            R += [r]
        Q = torch.stack(Q).view(batches + Q[0].shape)
        R = torch.stack(R).view(batches + R[0].shape)
        result = Q, R
    return result


@torch.jit.script
def sym(x):  # pragma: no cover
    return 0.5 * (x.transpose(-1, -2) + x)


@torch.jit.script
def extract_diag(x):  # pragma: no cover
    n, m = x.shape[-2:]
    batch = x.shape[:-2]
    k = n if n < m else m
    idx = torch.arange(k, dtype=torch.long, device=x.device)
    # torch script does not support Ellipsis indexing
    x = x.view(-1, n, m)
    return x[:, idx, idx].view(batch + (k,))


@torch.jit.script
def matrix_rank(x):  # pragma: no cover
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        result = torch.matrix_rank(x)
    else:
        batches = x.shape[:-2]
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        ranks = []
        # I wish I had a parallel_for
        for i in range(flat.shape[0]):
            r = torch.matrix_rank(slices[i])
            # interesting,
            # ranks.append(r)
            # does not work on pytorch 1.0.0
            # but the below code does
            ranks += [r]
        result = torch.stack(ranks).view(batches)
    return result


@torch.jit.script
def expm(x):  # pragma: no cover
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        result = _expm.expm_one(x)
    else:
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        exp = []
        # I wish I had a parallel_for
        for i in range(flat.shape[0]):
            e = _expm.expm_one(slices[i])
            exp += [e]
        result = torch.stack(exp).view(x.shape)
    return result
