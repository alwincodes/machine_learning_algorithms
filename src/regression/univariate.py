import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    #load the data we are using pandas
    file_location = "../../Data/Salary_Data.csv"
    df = pd.read_csv(file_location)

    """
    split data into X and y. Here we are not using testing and cross validation sets because we have limited data 
    """

    X = df.iloc[:, :1].values
    y = df.iloc[:, 1:].values

    #we will plot our data to see what we are working with

    plt.title("Salary / Experience")
    plt.scatter(X, y, s=10)
    plt.show()

    """
    Now we will have to fit a line which best represents the features in this dataset. to do this we use gradient descent, 
    first we will have to define a cost function and a derivative to the cost function with respect to theta value 
    (ie slope of different theta values) in univariate linear regression we have 2 theta values theta 1 and theta 2 
    y = theta1 + theta2 * X (we need to find the value of theta1 and theta2 which minimizes the cost function)
    """
    # add bias to X array
    ones = np.ones(len(X))
    X = np.insert(X, 0, ones, axis= 1)

    #cost calculation function
    def costCalculator(X, y, Theta):
        J = np.sum(np.power((np.matmul(X, Theta) - y), 2))

if __name__ == "__main__":
    main()

