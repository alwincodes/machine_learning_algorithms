from tools.preprocessing.normalize import multi_col_normalize, normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tools.algorithms.cost_function import linear_regression_cost

from tools.gradient_descent.batch_gd import batch_gradient_descent
from tools.graphs.learningvisualizer import plot_graphs
from tools.preprocessing.polynomial_features import generate_features

def main():
    #load the data we are using pandas
    file_location = "./Data/Position_Salaries.csv"
    df = pd.read_csv(file_location)

    """
    split data into X and y. Here we are not using testing and cross validation sets because we have limited data 
    """

    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1:].values

    #lets graph the data before working on it
    plt.title("Position_Salaries")
    plt.xlabel("Level")
    plt.ylabel("Salary")
    plt.scatter(X, y, s=5)
    plt.show()

    """
    As we can see from the graph the data is in a polynomial curve form so a straight line will not be able to fit it well. lets demonstrate
    this by trying linear regression with this dataset
    """

    # add bias to X array
    ones = np.ones(len(X))
    X = np.insert(X, 0, ones, axis= 1)

    #train the algorithm to get optimal theta values which minimize J(theta)
    init_theta = np.zeros(X.shape[1])
    alpha = 0.04
    iteration_count = 500
    theta, cost_per_iter = batch_gradient_descent(linear_regression_cost, alpha, X, y, init_theta, itcount=iteration_count)
    predictions = np.matmul(X, theta)

    #cost calculated for the value of theta is
    print(f"The cost calculated for value of Theta after {iteration_count} iteration is {cost_per_iter[iteration_count-1]}")
    #visualize our learning model
    plot_graphs(X[:, 1], y, predictions, cost_per_iter, xlabel="Level", ylabel="Salary")

    """
    As we can see in the graph our model tried its best to fit a straigt line through the data but it was not enough so lets try adding
    polynomial features
    """

    X = generate_features(X, 4) #generates polynomial features of 4 rows
    #since the generated polynomial features are very large we need to apply normalization
    y = normalize(y)
    # X[:, 1] = normalize(X[:, 1])
    # X[:, 2] = normalize(X[:, 2])
    # X[:, 3] = normalize(X[:, 3])
    # X[:, 4] = normalize(X[:, 4])
    X[:, 1:] = multi_col_normalize(X[:, 1:])
    init_theta = np.zeros(X.shape[1])
    theta, cost_per_iter = batch_gradient_descent(linear_regression_cost, alpha, X, y, init_theta, itcount=iteration_count)
    predictions = np.matmul(X, theta)
    #cost calculated for the value of theta is
    print(f"The cost calculated for value of Theta after {iteration_count} iteration is {cost_per_iter[iteration_count-1]}")
    #visualize our learning model
    plot_graphs(X[:, 1], y, predictions, cost_per_iter, xlabel="Level", ylabel="Salary")
   

if __name__ == "__main__":
    main()