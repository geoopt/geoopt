import torch

_TORCH_LESS_THAN_ONE = tuple(map(int, torch.__version__.split(".")[:2])) < (1, 0)
