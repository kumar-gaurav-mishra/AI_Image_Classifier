# Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
np.random.seed(1)

# Load datasets
train_x_orig, train_y, test_x_orig, test_y, classes = load_datasets()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255
test_x = test_x_flatten/255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# N Layer model
def N_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=10000, print_cost=False):
    np.random.seed(1)
    costs = []
    params = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = linear_model(X, params)
        cost = compute_cost(AL, Y)
        grads = activation_backward_model(AL, Y, caches)
        params = update_params(params, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return params

layer_dims = [12288, 20, 7, 5, 1]

params = N_layer_model(train_x, train_y, layer_dims,learning_rate=0.0075, num_iterations = 2500, print_cost = True)

# Actual image test
fileName = 'cat.png'
myImage = Image.open(fileName).convert('RGB').resize([num_px, num_px])
image = np.array(myImage)
test_image = image.reshape(num_px*num_px*3, 1)
test_image = test_image/255
prediction = predict(test_image, [1], params)

print ("y = " + str(np.squeeze(prediction)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(prediction)),].decode("utf-8") +  "\" picture.")
plt.imshow(image)