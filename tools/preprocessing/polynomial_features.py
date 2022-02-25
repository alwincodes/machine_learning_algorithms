import numpy as np

def generate_features(X, d = 2):
    res = np.zeros(shape=(len(X), d+1))
    res[:, 0] = X[:, 0]
    for i in range(1, d+1):
        res[:, i] = np.power(X[:, 1], i)
    
    return res