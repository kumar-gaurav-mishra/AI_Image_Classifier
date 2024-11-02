# Imports
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Initialize the b and w for deep NN
def initialize_parameters(layer_dims):
    np.random.seed(1)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1]) #* 0.01
        params['b'+str(l)] = np.zeros((layer_dims[l], 1))

        # Verify the dimetions of the each layer W and b
        assert(params['W' + str(l)]).shape == (layer_dims[l], layer_dims[l-1])
        assert(params['b' + str(l)]).shape == (layer_dims[l], 1) 

    return params

# Activation functions
# Linear Activation Function
def linear_activation(A, W, b):
    Z = np.dot(W, A) + b

    #verify the output shape for proper activation
    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)
    return Z, cache

# Sigmoid Activation Function
def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

# Relu Activation Function
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

# Activation Function forward
def activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_activation(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    
    # Output verification
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

# Linear Model Function Which replicate the n layers and uses different activation function for different layers
def linear_model(X, params):
    caches = []
    A = X
    L = len(params) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, params['W' + str(l)], params['b'+str(l)], 'relu')
        caches.append(cache)
    AL, cache = activation_forward(A, params['W' + str(L)], params['b' + str(L)], 'sigmoid' )
    caches.append(cache)

    # Verify the output shape
    assert(AL.shape == (1, X.shape[1]))
    return AL, caches

# Conpute Cost (Cost Function)
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

# Backward Activation functions
# Linear Backward Activation Function
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# Sigmoid Backward Activation Function
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape)
    return dZ

# Relu Backward Activation Function
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

# Get the value of dA_prev, dW, db by reverse engineering the activation and then lenear backward
def activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# Now let's find the value of dA_prev, dW, db for each and every layer. To do that we need activation_backward_model
def activation_backward_model(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

# Update parameters with dW and db values to prepare W and b for next iteration.
def update_params(params, grads, learning_rate):
    L = len(params) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return params

# Load Datasets
def load_datasets():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Predicts the results of the L-layer neural network
def predict(X, y, params):
    m = X.shape[1]
    n = len(params) // 2
    p = np.zeros((1, m))

    probas, caches = linear_model(X, params)
    
    # convert probas in 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
        
    print("Accuracy: "  + str(np.sum((p == y)/m)*100))
    return p