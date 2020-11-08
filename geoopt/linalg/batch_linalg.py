import torch.jit
from . import _expm

__all__ = [
    "svd",
    "qr",
    "sym",
    "extract_diag",
    "matrix_rank",
    "expm",
    "block_matrix",
    "sym_funcm",
    "sym_expm",
    "sym_logm",
    "sym_sqrtm",
    "sym_invm",
    "sym_inv_sqrtm1",
    "sym_inv_sqrtm2",
]


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


@torch.jit.script
def block_matrix(blocks: List[List[torch.Tensor]], dim0: int = -2, dim1: int = -1):
    # [[A, B], [C, D]] ->
    # [AB]
    # [CD]
    hblocks = []
    for mats in blocks:
        hblocks.append(torch.cat(mats, dim=dim1))
    return torch.cat(hblocks, dim=dim0)


def sym_funcm(
    x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Apply function to symmetric matrix.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    func : Callable[[torch.Tensor], torch.Tensor]
        function to apply

    Returns
    -------
    torch.Tensor
        symmetric matrix with function applied to
    """
    e, v = torch.symeig(x, eigenvectors=True)
    return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)


def sym_expm(x: torch.Tensor, using_native=False) -> torch.Tensor:
    r"""Symmetric matrix exponent.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    using_native : bool, optional
        if using native matrix exponent `torch.matrix_exp`, by default False

    Returns
    -------
    torch.Tensor
        :math:`\exp(x)`
    """
    if using_native:
        return torch.matrix_exp(x)
    else:
        return sym_funcm(x, torch.exp)


def sym_logm(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric matrix logarithm.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`\log(x)`
    """
    return sym_funcm(x, torch.log)


def sym_sqrtm(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root .

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`x^{1/2}`
    """
    return sym_funcm(x, torch.sqrt)


def sym_invm(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix inverse.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`x^{-1}`
    """
    return sym_funcm(x, torch.reciprocal)


def sym_inv_sqrtm1(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix inverse square root.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    torch.Tensor
        :math:`x^{-1/2}`
    """
    return sym_funcm(x, lambda tensor: torch.reciprocal(torch.sqrt(tensor)))


def sym_inv_sqrtm2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric matrix inverse square root, with square root return also.

    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        :math:`x^{-1/2}`, :math:`x^{1/2}`
    """
    e, v = torch.symeig(x, eigenvectors=True)
    sqrt_e = torch.sqrt(e)
    inv_sqrt_e = 1 / sqrt_e
    return (
        v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
        v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
    )


# left here for convenience
qr = torch.qr

svd = torch.svd
