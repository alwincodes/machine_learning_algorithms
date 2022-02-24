from locale import normalize
from tools.gradient_descent.batch_gd import batch_gradient_descent
from tools.preprocessing.normalize import normalize
from tools.algorithms.cost_function import linear_regression_cost
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_graphs(X, y, prediction, cost_per_iter):
    _plot, axis = plt.subplots(1, 2, figsize= (10, 5))
    
    axis[0].set_title("Salary / Experience (normalized)")
    axis[0].set_xlabel("Experience")
    axis[0].set_ylabel("Salary")
    axis[0].scatter(X, y, s=10, label="Training data")
    axis[0].plot(X, prediction, color="Orange", label="Fitted line (prediction)")
    axis[0].legend()

    axis[1].set_title("Gradient descent")
    axis[1].set_ylabel("Cost")
    axis[1].set_xlabel("Iterations")
    axis[1].plot(cost_per_iter, label="Cost")
    axis[1].legend()

    plt.show()

def main():
    #load the data we are using pandas
    file_location = "./Data/Salary_Data.csv"
    df = pd.read_csv(file_location)

    """
    split data into X and y. Here we are not using testing and cross validation sets because we have limited data 
    """

    X = df.iloc[:, :1].values
    y = df.iloc[:, 1:].values

    #normalize X and y
    X = normalize(X)
    y = normalize(y)

    #we will plot our data to see what we are working with
    # plt.title("Salary / Experience")
    # plt.scatter(X, y, s=10)
    # plt.show()

    """
    Now we will have to fit a line which best represents the features in this dataset. to do this we use gradient descent, 
    first we will have to define a cost function and a derivative to the cost function with respect to theta value 
    (ie slope of different theta values) in univariate linear regression we have 2 theta values theta 1 and theta 2 
    y = theta1 + theta2 * X (we need to find the value of theta1 and theta2 which minimizes the cost function)
    """
    
    # add bias to X array
    ones = np.ones(len(X))
    X = np.insert(X, 0, ones, axis= 1)

    #train the algorithm to get optimal theta values which minimize J(theta)
    init_theta = np.array([0,  0])
    alpha = 0.09
    iteration_count = 500
    theta, cost_per_iter = batch_gradient_descent(linear_regression_cost, alpha, X, y, init_theta, itcount=iteration_count)
    predictions = np.matmul(X, theta)

    #cost calculated for the value of theta is
    print(f"The cost calculated for value of Theta after {iteration_count} iteration is {cost_per_iter[iteration_count-1]}")

    # np.array(prediction).reshape(y.shape)
    # y_pred = y_pred.reshape((len(y), 1))
    
    #plotting the results
    plot_graphs(X[:, 1], y, predictions, cost_per_iter)


if __name__ == "__main__":
    main()

