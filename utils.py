from typing import Sequence
from tensor import Tensor
import random

def _construct(shape: Sequence[int], val):
    if not shape: return val()
    return [_construct(shape[1:], val) for _ in range(shape[0])]

def zeros(shape: Sequence[int]) -> Tensor:
    return Tensor(_construct(shape, lambda:0))

def ones(shape: Sequence[int]) -> Tensor:
    return Tensor(_construct(shape, lambda:1))

def random_uniform(shape: Sequence[int], low=0.0, high=1.0, seed=None) -> Tensor:
    if seed is not None:
        random.seed(seed)
    return Tensor(_construct(shape, lambda: random.uniform(low, high)))