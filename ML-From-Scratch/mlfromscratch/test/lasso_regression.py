import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Import helper functions
from sklearn.datasets import make_regression
from mlfromscratch.supervised_learning import LassoRegression
from mlfromscratch.utils import normalize, mean_squared_error
from mlfromscratch.utils import train_test_split, Plot

def main():

    X, y = make_regression(n_samples=1000, n_features=1, noise=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    model = LassoRegression(degree=15,
                            reg_factor=0.05,
                            learning_rate=0.001,
                            n_iterations=4000)
    model.fit(X_train, y_train)

    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label='Training Error')
    plt.legend(handles=[training])
    plt.title('Error Plot')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error: {} (given by reg_factor : {}'.format(mse,0.5))
    y_pred_line = model.predict(X)

    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Lasso Regression")
    plt.title("MSE: {}".format(mse, fontsize=10))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()

