import numpy as np
def batch_gradient_descent(costfn, alpha, X, y, theta, itcount = 1000):
    cost_per_iteration = np.zeros(itcount)
    for i in range(0, itcount):
        J, t_grad = costfn(X, y, theta)
        theta = theta - (alpha*t_grad)
        cost_per_iteration[i] = J

    return theta, cost_per_iteration
