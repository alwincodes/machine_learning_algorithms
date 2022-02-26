import numpy as np

def linear_regression_cost(X, y, Theta):
    m = len(y)
    t_length = len(Theta)
    pred = np.matmul(X, np.transpose([Theta]))
    error = np.subtract(pred, y)
    #cost function
    J = np.divide(np.sum(np.power(error, 2)), m)

    
    theta_grad = np.zeros(t_length)
    #iterative approach
    # for i in range(0, m):
    #     theta_grad += error[i] * X[i, :]

    #vectorized approach
    theta_grad = np.sum((error * X), axis=0)

    theta_grad = np.divide(theta_grad, m) 
    return J, theta_grad


def logistic_regression_cost(X, y, Theta):
    m = len(y)
    t_length = len(Theta)
    pred = np.matmul(X, np.transpose([Theta]))
    error = np.subtract(sigmoid_fn(pred), y) 
    #cost calculation
    J = sum(np.multiply(-y, np.log(sigmoid_fn(pred))) - np.multiply((1-y), np.log(1 - sigmoid_fn(pred))))
    J = J/m

    #gradient calculation vectorized approach
    theta_grad = np.sum((error * X), axis=0)
    theta_grad = np.divide(theta_grad, m) 
    return J, theta_grad


def sigmoid_fn(Z):
    return np.divide(1, (1+np.exp(-Z)))