
from __future__ import annotations
from typing import Sequence, Tuple, Union
import copy
import operator

Number = Union[int, float]


def _deepcopy(obj):
    return copy.deepcopy(obj)

def _infer_shape(data):
    if not isinstance(data, (list, tuple)):
        return () 
    if not data:
        return (0,) 
    
    first_shape = _infer_shape(data[0])
    for item in data[1:]:
        if _infer_shape(item) != first_shape:
            raise ValueError("Inconsistent nesting or dimension lengths.")
    return (len(data),) + first_shape

def _normalize_index(idx, rank):

    if not isinstance(idx, tuple):
        idx = (idx,)
    return idx + (slice(None),) * (rank - len(idx))

class Tensor:
    def __init__(self, data: Union[Number, Sequence]):
        if isinstance(data, Tensor):
            data = data.to_list()
        self._data = _deepcopy(data)
        self._shape = _infer_shape(self._data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def to_list(self):
        return _deepcopy(self._data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self._data})"

    def __getitem__(self, idx):
        idx = _normalize_index(idx, len(self.shape))
        
        def _get_recursive(data, index_tuple):
            if not index_tuple:
                return data
            
            head, *tail = index_tuple
            
            if isinstance(head, int):
                return _get_recursive(data[head], tail)
            elif isinstance(head, slice):
                return [_get_recursive(item, tail) for item in data[head]]

        result = _get_recursive(self._data, idx)
        
        if isinstance(result, (list, tuple)):
            return Tensor(result)
        else:
            return result

    def __setitem__(self, idx, value):

        idx = _normalize_index(idx, len(self.shape))
        
        def _set_recursive(data, index_tuple, val):

            head, *tail = index_tuple
            
            if not tail:
                if isinstance(head, int):
                    data[head] = val
                elif isinstance(head, slice):
                    sub_data = data[head]
                    val_list = val.to_list() if isinstance(val, Tensor) else val
                    for i in range(len(sub_data)):
                        sub_data[i] = val_list[i] if isinstance(val_list, list) else val_list
            else:
                if isinstance(head, int):
                    _set_recursive(data[head], tail, val)
                elif isinstance(head, slice):
                    sub_data = data[head]
                    val_list = val.to_list() if isinstance(val, Tensor) else val
                    for i in range(len(sub_data)):
                         _set_recursive(sub_data[i], tail, val_list[i] if isinstance(val_list, list) else val_list)

        _set_recursive(self._data, idx, value)

    def _elementwise_op(self, other, op):

        if not isinstance(other, (Number, Tensor)):
            return NotImplemented
        if isinstance(other, Tensor):
            assert self.shape == other.shape, "Tensors must have same shape for element-wise operations."
        
        def _apply_op_recursive(data, other_val):
    
            if isinstance(data, list):
                if isinstance(other_val, list):
                    return [ _apply_op_recursive(sub, other_sub) for sub, other_sub in zip(data, other_val) ]
                else:
                    return [ _apply_op_recursive(sub, other_val) for sub in data ]
            else:
                return op(data, other_val)
        
        other_data = other.to_list() if isinstance(other, Tensor) else other
        return Tensor(_apply_op_recursive(self._data, other_data))

    def __add__(self, other): return self._elementwise_op(other, operator.add)
    def __sub__(self, other): return self._elementwise_op(other, operator.sub)
    def __mul__(self, other): return self._elementwise_op(other, operator.mul)
    def __truediv__(self, other): return self._elementwise_op(other, operator.truediv)

    def copy(self, deep=False):
        return Tensor(_deepcopy(self._data)) if deep else Tensor(self._data)

class Matrix(Tensor):

    def __init__(self, data):
        super().__init__(data)
        if len(self.shape)!=2:
            raise ValueError("Matrix must be 2‑D")
    @property
    def rows(self): return self.shape[0]
    @property
    def cols(self): return self.shape[1]
    @property
    def T(self):
        from linalg import transpose
        return transpose(self)