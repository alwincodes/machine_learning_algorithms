import pandas as pd
import numpy as np
from tools.algorithms.cost_function import linear_regression_cost
from tools.gradient_descent.batch_gd import batch_gradient_descent
import matplotlib.pyplot as plt

def main():
    #load the data we are using pandas
    file_location = "./Data/Position_Salaries.csv"
    df = pd.read_csv(file_location)

    """
    split data into X and y. Here we are not using testing and cross validation sets because we have limited data 
    """

    #one hot encoding to change categorical data to new features of binary values
    df = pd.get_dummies(df, columns=["Position"])
    
    df = df[np.concatenate((df.columns[2:],df.columns[:2]))]
    

    #test-train split
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

 
    #adding bias array to X
    ones = np.ones(len(X))
    X = np.insert(X, 0, ones, axis= 1)


    #train the algorithm to get optimal theta values which minimize J(theta)
    init_theta = np.zeros(X.shape[1])
    alpha = 0.02
    iteration_count = 10000
    theta, cost_per_iter = batch_gradient_descent(linear_regression_cost, alpha, X, y, init_theta, itcount=iteration_count)
    predictions = np.matmul(X, theta)
    print(f"The cost calculated for value of Theta after {iteration_count} iteration is {cost_per_iter[iteration_count-1]}")
    
    plt.plot(cost_per_iter)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()




if __name__ == "__main__":
    main()