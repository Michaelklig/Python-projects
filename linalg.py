
from tensor import Tensor, Matrix, Number
from utils import zeros
import math
from typing import Union


def dot(a: Tensor, b: Tensor) -> "Union[Tensor, Number]":
    a_rank, b_rank = len(a.shape), len(b.shape)

    if (a_rank == 1 and a.shape[0] == 0) or (b_rank == 1 and b.shape[0] == 0):
        raise ValueError("Error: Empty vectors")
    if (a_rank == 2 and (a.shape[0] == 0 or a.shape[1] == 0)) or (b_rank == 2 and (b.shape[0] == 0 or b.shape[1] == 0)):
        raise ValueError("Error: Empty matrices.")

    if a_rank == 1 and b_rank == 1:  # Vector · Vector
        assert a.shape[0] == b.shape[0], "Vectors must have the same length."
        total = 0
        # TODO: Loop through elements, multiply and add to 'total'.
        for i in range(a.shape[0]):
            total += a[i] * b[i]
        return total

    elif a_rank == 2 and b_rank == 1:  # Matrix · Vector
        assert a.shape[1] == b.shape[0], "Matrix columns must match vector length."
        result_vec = zeros((a.shape[0],))
        # TODO: Loop through rows of matrix 'a', calculate the dot product of each row
        for i in range(a.shape[0]):
            temp_sum = 0
            for j in range(a.shape[1]):
                temp_sum += a[i, j] * b[j]
            result_vec[i] = temp_sum
        return result_vec

    elif a_rank == 2 and b_rank == 2:  # Matrix · Matrix
        assert a.shape[1] == b.shape[0], "Left matrix columns must match right matrix rows."
        result_mat = zeros((a.shape[0], b.shape[1]))
        # TODO: Loop through the rows of 'a' and columns of 'b'. For each position (i, j),
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                temp_sum = 0
                for k in range(a.shape[1]):
                    temp_sum += a[i, k] * b[k, j]
                result_mat[i, j] = temp_sum
        return result_mat
    else:
        raise NotImplementedError(
            "Dot product is only supported for vector/matrix combinations.")


def transpose(matrix: Union[Tensor, Matrix]) -> Union[Tensor, Matrix]:

    if len(matrix.shape) == 0:
        raise ValueError("Cannot transpose a scalar 0-dimensional.")

    if len(matrix.shape) == 1:
        matrix_len = matrix.shape[0]
        result = zeros((matrix_len, 1))
        # TODO: Handle the vector case: convert shape (n,) to (n, 1)
        if matrix_len == 0:
            return zeros((0, 1))
        for i in range(matrix_len):
            result[i, 0] = matrix[i]
        return result
    else:
        rows, cols = matrix.shape if len(
            matrix.shape) == 2 else (matrix.shape[0], 1)
        transposed_matrix = zeros((cols, rows))
        # TODO: Loop through the original matrix and assign each element matrix[i, j]
        for i in range(rows):
            for j in range(cols):
                transposed_matrix[j, i] = matrix[i, j]

        if isinstance(matrix, Matrix):
            return Matrix(transposed_matrix)
        else:
            return transposed_matrix


def mean_axis(tensor: Union[Tensor, Matrix], axis: int) -> Tensor:
    if len(tensor.shape) == 0:
        raise ValueError("Not defined for scalars.")
    if len(tensor.shape) > 2:
        raise AssertionError("Only implemented for rank 1 or 2 .")
    if len(tensor.shape) == 1:
        assert axis == 0, "For vectors, axis must be 0."
        vector_len = tensor.shape[0]
        if vector_len == 0:
            raise ValueError("Not defined for empty vectors.")
        total = 0
        for i in range(vector_len):
            total += tensor[i]
        return Tensor(total / vector_len)
    else:
        assert len(
            tensor.shape) == 2, "mean_axis is only implemented for rank-1 or rank-2 tensors."
        rows, cols = tensor.shape
        if rows == 0 or cols == 0:
            raise ValueError("Not defined for empty matrices.")
        if axis == 0:
            result_vec = zeros((cols,))
            for j in range(cols):
                col_sum = 0
                for i in range(rows):
                    col_sum += tensor[i, j]
                result_vec[j] = col_sum / rows
            return result_vec
        elif axis == 1:
            result_vec = zeros((rows,))
            for i in range(rows):
                row_sum = 0
                for j in range(cols):
                    row_sum += tensor[i, j]
                result_vec[i] = row_sum / cols
            return result_vec
        else:
            raise ValueError("Axis must be 0 or 1 for a matrix.")


def norm(tensor: Union[Tensor, Matrix], p: int = 2) -> Number:
    if len(tensor.shape) > 2:
        raise AssertionError("Only defined for vectors or matrices.")
    if len(tensor.shape) == 1:  # Vector norm
        length = tensor.shape[0]
        if length == 0:
            return 0
        total = 0
        if p == 1:  # L1 Norm
            for i in range(length):
                total += abs(tensor[i])
            return total
        elif p == 2:  # L2 Norm
            for i in range(length):
                total += tensor[i] ** 2
            return math.sqrt(total)
        else:
            raise ValueError("Norm is only defined for p=1 or p=2.")
    elif len(tensor.shape) == 2:
        row_count, col_count = tensor.shape
        if row_count == 0 or col_count == 0:
            return 0
        total = 0
        if p == 1:
            for i in range(row_count):
                for j in range(col_count):
                    total += abs(tensor[i, j])
            return total
        elif p == 2:
            for i in range(row_count):
                for j in range(col_count):
                    total += tensor[i, j] ** 2
            return math.sqrt(total)
        else:
            raise ValueError("Norm is only defined for p=1 or p=2.")
        return total
    else:
        raise AssertionError("Norm is only defined for vectors or matrices.")


def relu(tensor: Tensor) -> Tensor:
    if len(tensor.shape) == 0:
        return Tensor(max(0, tensor.to_list()))
    if len(tensor.shape) == 1 and tensor.shape[0] == 0:
        return Tensor([])
    if len(tensor.shape) == 2 and (tensor.shape[0] == 0 or tensor.shape[1] == 0):
        return Tensor([[]])
    result = zeros(tensor.shape)
    if len(tensor.shape) == 1:
        for i in range(tensor.shape[0]):
            result[i] = max(0, tensor[i])
    elif len(tensor.shape) == 2:
        rows, cols = tensor.shape
        for i in range(rows):
            for j in range(cols):
                result[i, j] = max(0, tensor[i, j])
    return result
