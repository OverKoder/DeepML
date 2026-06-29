import numpy as np
from typing import Tuple
import math

# Problem 1
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    if len(a[0]) != len(b):
        return -1

    # Reserve memory
    result = [0] * len(a)

    for i in range(len(a)):
        for j in range(len(b)):
            print(i, j, a[i][j], b[j])
            result[i] += a[i][j] * b[j]
	
    return result

# Problem 2
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """

    # We reserve memory and return a copy, the operation is not done inplace for simplicity
    new_a = [[0 for _ in range(len(a))] for _ in range(len(a[0]))]

    # Main loop
    for i in range(len(a)):
        for j in range(len(a[0])):
            new_a[j][i] = a[i][j]

    return new_a

# Problem 3
def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:

    # Check that the new_shape has the same number of elements as the current matrix
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return []
    
    new_a = [[0 for _ in range(new_shape[1])] for _ in range(new_shape[0])]
    new_i, new_j = 0, 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            new_a[new_i][new_j] = a[i][j]
            new_j += 1

            if new_j == new_shape[1]:
                new_j = 0
                new_i += 1

    return new_a

# Problem 4
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

    if mode == "row":
        means = [sum(matrix[i][:]) / len(matrix[0]) for i in range(len(matrix))]
    elif mode == "column":
        means = [sum([row[i] for row in matrix]) / len(matrix) for i in range(len(matrix[0]))]

    return means

# Problem 5
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	return [[elem * scalar for elem in row] for row in matrix]

# Problem 8
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.

    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]

    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    det_matrix = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    if det_matrix == 0:
        return None
    
    matrix_inverse = [[matrix[1][1], - matrix[0][1]], [- matrix[1][0], matrix[0][0]]]
    matrix_inverse = [[(1 / det_matrix) * elem for elem in row] for row in matrix_inverse]
    return matrix_inverse

# Problem 10
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	"""Write a Python function to calculate the covariance matrix for a given set of vectors.
	The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists.
	"""
	def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
		# From problem 4
		if mode == "row":
			means = [sum(matrix[i][:]) / len(matrix[0]) for i in range(len(matrix))]
		elif mode == "column":
			means = [sum([row[i] for row in matrix]) / len(matrix) for i in range(len(matrix[0]))]

		return means
	
	def covariance(a: list[int|float], b: list[int|float], mean_a: float, mean_b: float) -> int | float:
		# We assume len(a) == len(b)
		cov = 0
		for i in range(len(a)):
			cov += (a[i] - mean_a) * (b[i] - mean_b)
		
		return cov / (len(a) - 1)


	means = calculate_matrix_mean(vectors, mode = "row")
	covariance_matrix = [[0 for _ in range(len(vectors))] for _ in range(len(vectors))]

	for i in range(len(vectors)):
		for j in range(i, len(vectors)):
			covariance_matrix[i][j] = covariance(vectors[i], vectors[j], means[i], means[j])
			covariance_matrix[j][i] = covariance_matrix[i][j]

	return covariance_matrix

# Problem 14
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    theta = np.dot(
        np.dot( # (X^T * X)^-1 * X^T
            np.linalg.inv( # (X^T * X)^-1
                np.dot( # X^T * X
                    np.transpose(X), np.array(X) 
                )
            ),
            np.transpose(X)
        ),
    np.array(y)
    )
    return [round(elem ,4) for elem in theta.tolist()]

# Problem 16
def feature_scaling(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return standardized_data.tolist(), normalized_data.tolist()

# Problem 23
def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    softmax_scores = [math.pow(math.e, scores[i] - max_score) / sum([math.pow(math.e, scores[j] - max_score) for j in range(len(scores))])for i in range(len(scores))]
    return softmax_scores

# Problem 1049
def tanh_soft_cap(logits, softcap):
    """Apply tanh soft-capping to logits.

    Args:
        logits: numpy array of any shape.
        softcap: positive float, or None/<=0 to disable.

    Returns:
        numpy array of the same shape with soft-capped values,
        rounded to 6 decimal places.
    """
    if softcap is None or softcap <= 0:
        return logits

    return np.round(softcap * np.tanh(logits / softcap), 6)