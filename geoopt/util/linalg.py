import torch

__all__ = ["svd", "qr", "sym", "extract_diag", "matrix_rank"]


@torch.jit.script
def svd(x):
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        # 17 milliseconds on my mac to check that condition, that is low overhead
        return torch.svd(x)
    else:
        batches = x.shape[:-2]
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        U, D, V = [], [], []
        # I wish I had a parallel_for
        for i in range(flat.shape[0]):
            u, d, v = torch.svd(slices[i])
            U.append(u)
            D.append(d)
            V.append(v)
        U = torch.stack(U).view(batches + U[0].shape)
        D = torch.stack(D).view(batches + D[0].shape)
        V = torch.stack(V).view(batches + V[0].shape)
        return U, D, V


@torch.jit.script
def qr(x):
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        return torch.qr(x)
    else:
        batches = x.shape[:-2]
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        Q, R = [], []
        # I wish I had a parallel_for
        for i in range(flat.shape[0]):
            q, r = torch.qr(slices[i])
            Q.append(q)
            R.append(r)
        Q = torch.stack(Q).view(batches + Q[0].shape)
        R = torch.stack(R).view(batches + R[0].shape)
        return Q, R


@torch.jit.script
def sym(x):
    return 0.5 * (x.transpose(-1, -2) + x)


@torch.jit.script
def extract_diag(x):
    n, m = x.shape[-2:]
    batch = x.shape[:-2]
    k = n if n < m else m
    idx = torch.arange(k, dtype=torch.long, device=x.device)
    x = x.view(-1, n, m)
    return x[:, idx, idx].view(batch + (k,))


@torch.jit.script
def matrix_rank(x):
    # inspired by
    # https://discuss.pytorch.org/t/multidimensional-svd/4366/2
    # prolonged here:
    if x.dim() == 2:
        return torch.matrix_rank(x)
    else:
        batches = x.shape[:-2]
        other = x.shape[-2:]
        flat = x.view((-1,) + other)
        slices = flat.unbind(0)
        ranks = []
        # I with I had a parallel_for
        for i in range(flat.shape[0]):
            r = torch.matrix_rank(slices[i])
            ranks.append(r)
        ranks = torch.stack(ranks).view(batches)
        return ranks
