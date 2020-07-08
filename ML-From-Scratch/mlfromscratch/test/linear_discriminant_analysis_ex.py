from __future__ import print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from mlfromscratch.supervised_learning import LDA
from mlfromscratch.utils import calculate_covariance_matrix, accuracy_score
from mlfromscratch.utils import train_test_split, Plot

def main():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(X_test, y_pred)
    print("Accuracy : {}".format(accuracy))
    Plot().plot_in_2d(X_test, y_test,title="LDA", accuracy=accuracy)

if __name__ == "__main__":
    main()
