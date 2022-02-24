from locale import normalize
from tools.gradient_descent.batch_gd import batch_gradient_descent
from tools.preprocessing.normalize import normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

    #cost and gradient calculation function
    def costCalculator(X, y, Theta):
        m = len(y)
        t_length = len(Theta)
        pred = np.matmul(X, np.transpose([Theta]))
        #cost function
        J = np.sum(np.power((pred - y), 2)) * 1/m

        #implement gradient fn later
        theta_grad = np.zeros(t_length)
        for i in range(0, m):
            theta_grad += (pred[i] - y[i]) * X[i, :]

        theta_grad /= m
        return J, theta_grad

    init_theta = np.array([0,  0])
    theta, cost_per_iter = batch_gradient_descent(costCalculator, 0.07, X, y, init_theta)
    prediction = np.matmul(X, theta)
    y_pred = np.array(prediction)
    y_pred = y_pred.reshape((len(y), 1))
    
    plt.title("Salary / Experience")
    plt.scatter(X[:, 1], y, s=10)
    plt.plot(X[:, 1], prediction)
    plt.show()

    plt.title("Gradient descent")
    plt.plot(cost_per_iter)
    plt.show()

if __name__ == "__main__":
    main()

