import math
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
    epsilon = 1e-20  #small value to avoid log(0)
    m = len(y)
    t_length = len(Theta)
    pred = np.matmul(X, np.transpose([Theta]), dtype=np.float64)
    sigarr = sigmoid_fn(pred)
    error = np.subtract(sigarr, y)
    #cost calculation
    J = sum(np.multiply(-1* y, np.log(sigarr+ epsilon)) - np.multiply((1-y), np.log(1 - sigarr + epsilon)))
    J = J/m

    #gradient calculation vectorized approach
    theta_grad = np.sum((error * X), axis=0)
    theta_grad = theta_grad/m
    return J, theta_grad


def sigmoid_fn(x):
    """
    A numerically stable version of the logistic sigmoid function.
    I dont know why this works but I have tried to implement it myself but was having overflow erros :(
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)