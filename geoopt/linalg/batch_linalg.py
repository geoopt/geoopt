from typing import Tuple
import torch.jit
from . import _expm

__all__ = ["svd", "qr", "sym", "extract_diag", "matrix_rank", "expm", "block_matrix"]


@torch.jit.script
def sym(x: torch.Tensor):  # pragma: no cover
    return 0.5 * (x.transpose(-1, -2) + x)


@torch.jit.script
def extract_diag(x: torch.Tensor):  # pragma: no cover
    n, m = x.shape[-2:]
    batch = x.shape[:-2]
    k = n if n < m else m
    idx = torch.arange(k, dtype=torch.long, device=x.device)
    # torch script does not support Ellipsis indexing
    x = x.view(-1, n, m)
    return x[:, idx, idx].view(batch + (k,))


@torch.jit.script
def matrix_rank(x: torch.Tensor):  # pragma: no cover
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
def expm(x: torch.Tensor):  # pragma: no cover
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


def block_matrix(blocks: Tuple[Tuple[torch.Tensor, ...], ...]):
    # [[A, B], [C, D]] ->
    # [AB]
    # [CD]
    blocks = tuple(torch.cat(mats, dim=-1) for mats in blocks)
    return torch.cat(blocks, dim=-2)


# left here for convenience
qr = torch.qr

svd = torch.svd
