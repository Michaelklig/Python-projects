import unittest
import math
from tensor import Tensor, Matrix
from utils import zeros
from linalg import dot, transpose, mean_axis, norm, relu


class TestCoreFunctions(unittest.TestCase):
    def test_dot_vector_vector(self):
        self.assertEqual(dot(Tensor([1, 2, 3]), Tensor([4, 5, 6])), 32)

    def test_dot_matrix_vector(self):
        m, v = Tensor([[1, 2], [3, 4]]), Tensor([5, 6])
        self.assertEqual(dot(m, v).to_list(), [17, 39])

    def test_transpose_square(self):
        m = Tensor([[1, 2], [3, 4]])
        self.assertEqual(transpose(m).to_list(), [[1, 3], [2, 4]])

    def test_transpose_rectangular(self):
        m = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(transpose(m).shape, (3, 2))

    def test_transpose_vector(self):
        v = Tensor([1, 2, 3])
        t = transpose(v)
        self.assertEqual(t.shape, (3, 1))
        self.assertEqual(t.to_list(), [[1], [2], [3]])
        self.assertIsInstance(t, Tensor)

    def test_transpose_matrix_type(self):
        m = Matrix([[1, 2], [3, 4]])
        t = transpose(m)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.to_list(), [[1, 3], [2, 4]])
        self.assertIsInstance(t, Matrix)

    def test_mean_axis_0(self):
        m = Tensor([[1, 10], [3, 12], [5, 14]])
        self.assertTrue(math.isclose(mean_axis(m, axis=0)[1], 12.0))

    def test_mean_axis_1(self):
        m = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(mean_axis(m, axis=1).to_list(), [2.0, 5.0])

    def test_mean_axis_vector(self):
        v = Tensor([1, 2, 3, 4])
        mean = mean_axis(v, axis=0)
        # Should be a scalar Tensor or 0-dim Tensor
        self.assertTrue(abs(mean.to_list() - 2.5) <
                        1e-8 or abs(mean[()] - 2.5) < 1e-8)

    def test_mean_axis_matrix_axis0(self):
        m = Matrix([[1, 2], [3, 4]])
        mean = mean_axis(m, axis=0)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(mean.to_list(), [2.0, 3.0])

    def test_mean_axis_matrix_axis1(self):
        m = Matrix([[1, 2], [3, 4]])
        mean = mean_axis(m, axis=1)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(mean.to_list(), [1.5, 3.5])

    def test_norm_l1(self):
        v = Tensor([1, -2, 3])
        self.assertEqual(norm(v, p=1), 6)

    def test_norm_l2(self):
        v = Tensor([3, 4])
        self.assertEqual(norm(v, p=2), 5)

    def test_norm_matrix_l1(self):
        m = Matrix([[1, -2], [3, -4]])
        self.assertEqual(norm(m, p=1), 10)

    def test_norm_matrix_l2(self):
        m = Matrix([[3, 4], [0, 0]])
        self.assertEqual(norm(m, p=2), 5)

   # Extra Edge Cases ------------------------------------------------------------------------------------
    # raises ValueError when vectors are empty
    def test_empty_vector(self):
        v = Tensor([])
        with self.assertRaises(ValueError):
            dot(v, v)

    # raises AssertionError when vector lengths don't match
    def test_shape_mismatch(self):
        v1 = Tensor([1, 2, 3])
        v2 = Tensor([1, 2])
        with self.assertRaises(AssertionError):
            dot(v1, v2)

    # Ensure transposing a scalar raises ValueError
    def test_transpose_scalar(self):
        s = Tensor(5)
        with self.assertRaises(ValueError):
            transpose(s)

    # raises ValueError when applied to an empty vector
    def test_mean_axis_empty_vector(self):
        v = Tensor([])
        with self.assertRaises(ValueError):
            mean_axis(v, axis=0)

    # raises ValueError when applied to an empty matrix
    def test_mean_axis_empty_matrix(self):
        m = Tensor([[]])
        with self.assertRaises(ValueError):
            mean_axis(m, axis=0)

    # raises ValueError when axis is out of bounds
    def test_mean_axis_invalid_axis(self):
        m = Tensor([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            mean_axis(m, axis=2)

    # Test norm of empty vector returns 0 for both L1 and L2
    def test_norm_empty_vector(self):
        v = Tensor([])
        self.assertEqual(norm(v, p=1), 0)
        self.assertEqual(norm(v, p=2), 0)

    # Test norm of empty matrix returns 0 for both L1 and L2
    def test_norm_empty_matrix(self):
        m = Tensor([[]])
        self.assertEqual(norm(m, p=1), 0)
        self.assertEqual(norm(m, p=2), 0)

    # raises ValueError when given an unsupported norm type (p=3)
    def test_norm_invalid_p(self):
        v = Tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            norm(v, p=3)

    # Raises NotImplementedError when dot is called on tensors with rank > 2
    def test_dot_higher_rank_tensor(self):
        t = Tensor([[[1, 2], [3, 4]]])  # shape (1, 2, 2)
        with self.assertRaises(NotImplementedError):
            dot(t, t)

    # Raises AssertionError or NotImplementedError for mean_axis on rank > 2
    def test_mean_axis_higher_rank_tensor(self):
        t = Tensor([[[1, 2], [3, 4]]])
        with self.assertRaises(AssertionError):
            mean_axis(t, axis=0)

    # Raises AssertionError or NotImplementedError for norm on rank > 2
    def test_norm_higher_rank_tensor(self):
        t = Tensor([[[1, 2], [3, 4]]])
        with self.assertRaises(AssertionError):
            norm(t)


class TestBonusQuestion(unittest.TestCase):
    def test_relu_basic(self):
        t = Tensor([-10, 0, 10])
        self.assertEqual(relu(t).to_list(), [0, 0, 10])

    def test_relu_on_matrix(self):
        m = Tensor([[-1, -2], [3, 0]])
        self.assertEqual(relu(m).to_list(), [[0, 0], [3, 0]])

    # Verify that applying ReLU to an empty vector and returns an empty list
    def test_relu_empty_vector(self):
        v = Tensor([])
        self.assertEqual(relu(v).to_list(), [])

    # Verify that applying ReLU to an empty matrix returns a matrix with the same structure
    def test_relu_empty_matrix(self):
        m = Tensor([[]])
        self.assertEqual(relu(m).to_list(), [[]])

    # returns 0 when applied to a negative scalar
    def test_relu_scalar_negative(self):
        s = Tensor(-5)
        self.assertEqual(relu(s).to_list(), 0)

    # returns the same positive scalar when input > 0
    def test_relu_scalar_positive(self):
        s = Tensor(7)
        self.assertEqual(relu(s).to_list(), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
