__all__ = "copy_or_set_"


def copy_or_set_(dest, source):
    """
    Copy or inplace set from :code:`source` to :code:`dest`.

    A workaround to respect strides of :code:`dest` when copying :code:`source`.
    The original issue was raised `here <https://github.com/geoopt/geoopt/issues/70>`_
    when working with matrix manifolds. Inplace set operation is mode efficient,
    but the resulting storage might be incompatible after. To avoid the issue we refer to
    the safe option and use :code:`copy_` if strides do not match.

    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor

    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


def strip_tuple(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def make_tuple(obj):
    if not isinstance(obj, tuple):
        return (obj,)
    else:
        return obj


def size2shape(*size):
    return make_tuple(strip_tuple(size))
