import numpy as np

def linear_regression_cost(X, y, Theta):
    m = len(y)
    t_length = len(Theta)
    pred = np.matmul(X, np.transpose([Theta]))
    error = pred - y
    #cost function
    J = np.sum(np.power(error, 2)) * 1/m

    
    theta_grad = np.zeros(t_length)
    #iterative approach
    # for i in range(0, m):
    #     theta_grad += error[i] * X[i, :]

    #vectorized approach
    theta_grad = np.sum((error * X), axis=0)

    theta_grad /= m
    return J, theta_grad

