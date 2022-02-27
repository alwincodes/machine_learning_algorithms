import idx2numpy
import numpy as np


#the file names where the mnist data is stored
file_train_images = 'Data/mnist_dataset/train-images.idx3-ubyte'
file_train_label = 'Data/mnist_dataset/train-labels.idx1-ubyte'
file_test_images = 'Data/mnist_dataset/t10k-images.idx3-ubyte'
file_test_label = 'Data/mnist_dataset/t10k-labels.idx1-ubyte'

#since the file is in an idx format we use idx2numpy package to convert it into a usable numpy array
# train_images = idx2numpy.convert_from_file(file_train_images)
# train_labels = idx2numpy.convert_from_file(file_train_label)
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
X = test_images.astype(np.float64)
y = test_labels

#flatten the 60000 x 28 x 28 matrix into a 6000 x 784 matrix
X_test = X.reshape( (X.shape[0], X.shape[1]*X.shape[2])) 
#creating the bias 1 coloumn
ones = np.ones((X.shape[0]))
X_test = np.insert(X_test, 0, ones, axis=1)

theta = np.load("logistic_number_model.npy")

pred = np.matmul(X_test[1:10, :], theta.T)

print(np.argmax(pred, axis=0))
print(y[0:10])

