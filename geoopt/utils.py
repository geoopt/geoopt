import itertools
from typing import Tuple, Any, Union
import torch
import geoopt

__all__ = [
    "copy_or_set_",
    "strip_tuple",
    "size2shape",
    "make_tuple",
    "broadcast_shapes",
    "ismanifold",
    "canonical_manifold",
]


def copy_or_set_(dest: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
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


def strip_tuple(tup: Tuple) -> Union[Tuple, Any]:
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def make_tuple(obj: Union[Tuple, Any]) -> Tuple:
    if not isinstance(obj, tuple):
        return (obj,)
    else:
        return obj


def size2shape(*size: Union[Tuple[int], int]) -> Tuple[int]:
    return make_tuple(strip_tuple(size))


def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    """Apply numpy broadcasting rules to shapes."""
    result = []
    for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dim: int = 1
        for d in dims:
            if dim != 1 and d != 1 and d != dim:
                raise ValueError("Shapes can't be broadcasted")
            elif d > dim:
                dim = d
        result.append(dim)
    return tuple(reversed(result))


def ismanifold(instance, cls):
    """
    Check if interface of an instance is compatible with given class.

    Parameters
    ----------
    instance : geoopt.Manifold
        check if a given manifold is compatible with cls API
    cls : type
        manifold type

    Returns
    -------
    bool
        comparison result
    """
    if not issubclass(cls, geoopt.manifolds.Manifold):
        raise TypeError("`cls` should be a subclass of geoopt.manifolds.Manifold")
    if not isinstance(instance, geoopt.manifolds.Manifold):
        return False
    else:
        # this is the case to care about, Scaled class is a proxy, but fails instance checks
        while isinstance(instance, geoopt.Scaled):
            instance = instance.base
        return isinstance(instance, cls)


def canonical_manifold(manifold: "geoopt.Manifold"):
    """
    Get a canonical manifold.

    If a manifold is wrapped with Scaled. Some attributes may not be available. This should help if you really need them.

    Parameters
    ----------
    manifold : geoopt.Manifold

    Returns
    -------
    geoopt.Maniflold
        an unwrapped manifold
    """
    while isinstance(manifold, geoopt.Scaled):
        manifold = manifold.base
    return manifold
