from __future__ import division
import numpy as np
import math
import sys


def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def calculate_covariance_matrix(X, Y=None):
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1))*(X - X.mean(axis=0)).T.dot(X - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)

