import numpy as np


# Problem 30
def batch_iterator(X, y=None, batch_size=64):
    if y is not None:
        return [
            [X[idx : idx + batch_size].tolist(), y[idx : idx + batch_size].tolist()]
            for idx in range(0, len(X), batch_size)
        ]
    else:
        return [
            [X[idx : idx + batch_size].tolist()] for idx in range(0, len(X), batch_size)
        ]


# Problem 34
def to_categorical(x, n_col=None):

    # Create a one_hoy_matrix of zeros
    one_hot_matrix = (
        np.zeros((len(x), len(np.unique(x))))
        if n_col is None
        else np.zeros((len(x), n_col))
    )

    # We leverage the x vector as indices of where to put ones
    one_hot_matrix[np.arange(len(x)), x] = 1

    return one_hot_matrix


# Problem 36
def accuracy_score(y_true, y_pred):
    return np.sum(np.where(y_pred == y_true, 1, 0)) / y_true.shape[0]


# Problem 43
def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    ridge = (1 / X.shape[0]) * np.sum(
        np.square(np.dot(X, w) - y_true)
    ) + alpha * np.sum(np.square(w))
    return ridge


# Problem 45
def kernel_function(x1, x2):
    return np.dot(x1, np.transpose(x2))


# Problem 46
def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if (tp + fp) == 0:
        return 0.0

    return tp / (tp + fp)


# Problem 52
def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if (tp + fn) == 0:
        return 0.0

    return tp / (tp + fn)
