import math
import numpy as np


# Problem 6
def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

    a, b, c = (
        1,
        -(matrix[0][0] + matrix[1][1]),
        matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
    )
    eigen_0, eigen_1 = (
        (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a),
        (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a),
    )
    return [eigen_0, eigen_1] if eigen_0 >= eigen_1 else [eigen_1, eigen_0]


# Problem 7
def transform_matrix(
    A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]
) -> list[list[int | float]]:
    """Write a Python function that transforms a given matrix A using the operation T^-1 * A * S, where T and S are invertible matrices.
    The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1"""

    def matrix_multiplication(
        A: list[list[int | float]], B: list[list[int | float]]
    ) -> list[list[int | float]]:
        new_matrix = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

        # We assume here that columns(A) == rows(B)
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    new_matrix[i][j] += A[i][k] * B[k][j]

        return new_matrix

    det_T, det_S = np.linalg.det(T), np.linalg.det(S)

    if det_T == 0 or det_S == 0:
        return -1

    T_inverse = np.linalg.inv(T)

    result = matrix_multiplication(T_inverse, A)
    result = matrix_multiplication(result, S)
    return result
