#from __future__ import absolute_import
from __future__ import print_function, division
import numpy as np

from mlfromscratch.utils import calculate_covariance_matrix

class LDA():
    """THe Linear Discriminant Analysis also known as Fischer's linear discriminant.
    Besides classification it can also be used to reduce the dimensionality of the dataset.
    """
    def __init__(self):
        self.w = None

    def transform(self, X, y):
        self.fit(X, y)
        #project X onto w
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self, X, y):
        #separate the data by class
        X1 = X[y == 0]
        X2 = X[y == 1]

        #calculating the covariance matrix
        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        #calculate the mean
        mean1 = X1.mean(0)
        mean2 = X2.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        #calculating the vector w onto which the data will get projected
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
