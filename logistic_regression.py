"""
Goal of this project is to create a one vs all logistic regression for classifying handwritten digits using the mnist dataset

"""

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from tools.algorithms.cost_function import linear_regression_cost, logistic_regression_cost, sigmoid_fn
from tools.preprocessing.normalize import multi_col_normalize
from tools.gradient_descent.batch_gd import batch_gradient_descent

#the file names where the mnist data is stored
file_train_images = 'Data/mnist_dataset/train-images.idx3-ubyte'
file_train_label = 'Data/mnist_dataset/train-labels.idx1-ubyte'
file_test_images = 'Data/mnist_dataset/t10k-images.idx3-ubyte'
file_test_label = 'Data/mnist_dataset/t10k-labels.idx1-ubyte'

#since the file is in an idx format we use idx2numpy package to convert it into a usable numpy array
train_images = idx2numpy.convert_from_file(file_train_images)
train_labels = idx2numpy.convert_from_file(file_train_label)
test_images = idx2numpy.convert_from_file(file_test_images)
test_labels = idx2numpy.convert_from_file(file_test_label)

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# for i in range(0, 20):
#     print(train_labels[i])
#     plt.imshow(train_images[i], cmap="Greys")
#     plt.show()

#training set
X = train_images.astype(np.float64)
y = train_labels

#flatten the 60000 x 28 x 28 matrix into a 6000 x 784 matrix
X = X.reshape( (X.shape[0], X.shape[1]*X.shape[2])) 
# np.savetxt('data.csv', X, delimiter=',')
#reshape y to be a vector of 60000 dimenssions
y = y.reshape((y.shape[0], 1))
#creating the bias 1 coloumn
ones = np.ones((X.shape[0]))
X = np.insert(X, 0, ones, axis=1)



model = np.zeros((10, X.shape[1]))
# X[:, 1:] = multi_col_normalize(X[:, 1:])
#we store our models here

for i in range(0, 1):
    y_train  = (y == i).astype("int")

    #creating initial theta values for the coloumns
    init_theta = np.zeros(X.shape[1], dtype=np.float64)
    
    alpha = 0.01
    iteration_count = 10
    theta, cost_per_iter = batch_gradient_descent(logistic_regression_cost, alpha, X, y_train, init_theta, itcount=iteration_count)
    model[i, :] = theta
    predictions = sigmoid_fn(np.matmul(X, theta))
    print(f"The cost calculated for value of Theta after {iteration_count} iteration is {cost_per_iter[iteration_count-1]}")
    plt.plot(cost_per_iter)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
 

#save the theta array for further use
np.save("logistic_number_model", model)