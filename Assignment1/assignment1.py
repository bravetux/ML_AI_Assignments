"""
Multivariate Linear Regression example for advertising -> sales.

This script builds a linear regression model (closed-form normal equation)
using only NumPy. It trains on the provided dataset, evaluates RMSE and R^2,
prints coefficients and intercept, and demonstrates a sample prediction.

Run with the workspace Python. No external packages required beyond NumPy.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Return features X and target y as NumPy arrays.

    Dataset columns: TV, Radio, Newspaper, Sales
    """
    data_str = """
    230.1,37.8,69.2,22.1
    44.5,39.3,45.1,10.4
    17.2,45.9,69.3,9.3
    151.5,41.3,58.5,18.5
    180.8,10.8,58.4,12.9
    8.7,48.9,75.0,7.2
    57.5,32.8,23.5,11.8
    120.2,19.6,11.6,13.2
    144.1,16.0,40.3,15.6
    111.6,12.6,37.9,12.2
    """
    rows = [line.strip() for line in data_str.strip().splitlines() if line.strip()]
    arr = np.array([[float(x) for x in row.split(",")] for row in rows], dtype=float)
    X = arr[:, :3]  # TV, Radio, Newspaper
    y = arr[:, 3]   # Sales
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 1):
    """
    Split the dataset into a training set and a test set.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix to split.
    y : np.ndarray
        The target vector to split.
    test_ratio : float, optional
        The ratio of samples to use in the test set. Default is 0.2.
    seed : int, optional
        The seed to use for shuffling the samples. Default is 1.

    Returns
    -------
    X_train : np.ndarray
        The feature matrix for the training set.
    X_test : np.ndarray
        The feature matrix for the test set.
    y_train : np.ndarray
        The target vector for the training set.
    y_test : np.ndarray
        The target vector for the test set.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    test_size = max(1, int(round(n * test_ratio)))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def add_bias(X: np.ndarray) -> np.ndarray:
    """Add a bias term to the feature matrix X.

    The bias term is added as a column of ones to the feature matrix X.
    The resulting feature matrix is returned.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix to add a bias term to.

    Returns
    -------
    X_bias : np.ndarray
        The feature matrix with a bias term added.
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit linear regression using the normal equation and return theta vector.

    theta = (X^T X)^-1 X^T y
    """
    XtX = X.T @ X
    try:
        theta = np.linalg.inv(XtX) @ X.T @ y
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse if XtX is singular
        theta = np.linalg.pinv(XtX) @ X.T @ y
    return theta


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predict values using a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Features to make predictions on.
    theta : np.ndarray
        Model parameters (intercept and coefficients).

    Returns
    -------
    y_pred : np.ndarray
        Predicted values.
    """
    return X @ theta


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Calculate the root mean squared error between two arrays of numbers.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return math.sqrt(float(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R^2 score between two arrays of numbers.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        R^2 score.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def main() -> None:
    """
    Load dataset, train a linear regression model using normal equation,
    evaluate its performance on train and test sets, and demonstrate a prediction
    """
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

    Xb_train = add_bias(X_train)
    theta = fit_normal_equation(Xb_train, y_train)

    # Separate intercept and coefficients for readability
    intercept = float(theta[0])
    coefs = theta[1:]

    print("Learned model:")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Coefficients (TV, Radio, Newspaper): {[float(c) for c in coefs]}")

    # Evaluate on train and test
    Xb_test = add_bias(X_test)
    y_pred_test = predict(Xb_test, theta)
    y_pred_train = predict(Xb_train, theta)

    print("\nPerformance:")
    print(f"  Train RMSE: {rmse(y_train, y_pred_train):.4f}")
    print(f"  Train R^2 : {r2_score(y_train, y_pred_train):.4f}")
    print(f"  Test  RMSE: {rmse(y_test, y_pred_test):.4f}")
    print(f"  Test  R^2 : {r2_score(y_test, y_pred_test):.4f}")

    # Demonstrate a prediction
    sample = np.array([[150.0, 30.0, 40.0]])  # TV, Radio, Newspaper budgets
    sample_b = add_bias(sample)
    sample_pred = predict(sample_b, theta)[0]
    print("\nSample prediction:")
    print(f"  Budgets -> TV: {sample[0,0]}, Radio: {sample[0,1]}, Newspaper: {sample[0,2]}")
    print(f"  Predicted Sales (units): {sample_pred:.3f}")


if __name__ == "__main__":
    main()
