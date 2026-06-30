import math
import numpy as np
from typing import List


# Problem 22
def sigmoid(z: float) -> float:
    return 1 / (1 + math.pow(math.e, -z))


# Problem 24
def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> (List[float], float):
    def sigmoid(z: float):  # Problem 22
        return round(1 / (1 + math.pow(math.e, -z)), 4)

    def mse_loss(activations: np.ndarray, labels: np.ndarray):
        loss = (1 / activations.shape[0]) * np.sum(np.square(activations - labels))
        return round(loss, 4)

    activations = np.array([sigmoid(elem) for elem in np.dot(features, weights) + bias])
    mse = mse_loss(activations, labels)

    return activations.tolist(), mse.tolist()


# Problem 27
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    return np.round(np.dot(np.linalg.inv(C), B), 4).tolist()


# Problem 29
def shuffle_data(X, y, seed=None):
    indices = np.arange(len(X))

    # Shuffle indices
    np.random.seed(seed)
    np.random.shuffle(indices)
    # indices = np.random.choice(indices, len(X), replace=False) is also a solution

    # Shuffle X and y and return
    return X[indices], y[indices]


# Problem 39
def log_softmax(scores: list) -> np.ndarray:
    max_score = max(scores)
    log_softmax_scores = [
        score
        - max_score
        - np.log(
            np.sum([np.pow(np.e, scores[j] - max_score) for j in range(len(scores))])
        )
        for score in scores
    ]
    return np.array([round(elem, 4) for elem in log_softmax_scores])


# Problem 42
def relu(z: float) -> float:
    return z if z >= 0 else 0


# Problem 44
def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    return z if z >= 0 else alpha * z
