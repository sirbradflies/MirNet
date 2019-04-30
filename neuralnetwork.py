# TODO: Add Autoencoder option
# TODO: FIX REGRESSOR VS CLASSIFIER (PROBABILITY)
# TODO: AUTOMATIC NETWORK SETUP (META-NEURAL NETWORK)
# TODO: Memory optimization

import numpy as np
import time
import copy
import sklearn.metrics as mt

version = '71'
eps = 1e-9


# ReLu function
def relu(input_value, deriv=False):
    if deriv:
        return np.where(input_value < 0, 0, 1)
    else:
        return np.clip(input_value, 0, None)


# Logistic function
# TODO: Fix overflow with RELU
def logistic(input_value, deriv=False):
    output = 1 / (1 + np.exp(-input_value))
    if deriv:
        return output*(1-output)
    else:
        return output


# Identity function
def identity(input_value, deriv=False):
    if deriv:
        return 1
    else:
        return input_value


# Neural Network class
class NeuralNetwork(object):
    #  Create Neural Network with given parameters
    def __init__(self, hidden_layers=(100,), seed=None, type="classifier", activation="relu", verbose=False):
        self.seed = seed
        np.random.seed(self.seed)  # random generator initialization
        self.activation = activation
        self.hidden_act = globals().copy()[activation]
        self.type = type
        self.epochs = 0
        if type == "classifier":  # Classifier with multiple 2-class output neurons
            self.loss = mt.log_loss
            self.out_activation = logistic
        else:  # Regressor
            self.loss = mt.mean_squared_error
            self.out_activation = identity
        self.hidden_layers = hidden_layers
        self.weights = []
        self.verbose = verbose

    # Feed forward an input matrix through the Neural Network
    def feedForward(self, input, weights):
        neurons = [input]
        activations = [np.dot(neurons[0], weights[0])]
        for w in weights[1:]:  # Hidden layers feed forward
            neurons.append(self.hidden_act(activations[-1]))
            activations.append(np.dot(neurons[-1], w))
        # Output layer neurons calculation
        neurons.append(self.out_activation(activations[-1]))
        return neurons, activations # TODO: Activations needed?

    # Train the Neural Network through backpropagation
    # TODO: SGD Fraction
    def fit(self, X, Y, max_epochs=100, sgd_init=100, rate_init=0.001, sgd_annealing=0.9, rate_annealing=0.9,
            b1=0.1, b2=0.1, tolerance=0.01, X_val=None, Y_val=None, val_fraction=0.1, patience=10, max_time=0):
        layers = (X.shape[1],) + self.hidden_layers + (Y.shape[1],)
        depth = len(layers)
        vel, cache = [], []
        for (start, end) in zip(layers[:-1],layers[1:]):  # Start - End neurons number
            if len(self.weights) < depth-1:
                #self.weights.append(np.random.uniform(-0.5 / start,0.5 / start,(start, end)))  # assigning bounded weight
                self.weights.append(np.random.uniform(-0.5, 0.5,(start, end))
                                    *np.sqrt(2/start))  # HE et al. (2015) initialization TODO: Test with 0-1 random?
            vel.append(np.zeros((start, end)))  # initializing velocity values for network learning (Momentum and Adam)
            cache.append(np.zeros((start, end)))  # initializing cache values for network learning (Adagrad and Adam)

        weights_temp = copy.deepcopy(self.weights)

        learning_rate = rate_init
        self.train_losses, self.val_losses = [], []

        if X_val is None or Y_val is None:
            val_samples = int(X.shape[0] * val_fraction)
            shuffled_index = np.random.permutation(X.shape[0])
            val_index = shuffled_index[-val_samples:]
            X_val = X[val_index]
            Y_val = Y[val_index]
            train_index = shuffled_index[:-val_samples]
            x_train = X[train_index]
            y_train = Y[train_index]
        else:
            x_train, y_train = X, Y

        train_samples = x_train.shape[0]
        batch = min(sgd_init, train_samples)  # samples trained at the same time
        start_time = time.time()
        for epoch in range(1, max_epochs + 1):  # training the network
            if self.verbose:
                print("Starting Epoch: %i" % (epoch))
            folds = train_samples // batch  # iterations per epoch
            shuffled_index = np.random.permutation(train_samples)  # reshuffling of the train set
            x_train, y_train = x_train[shuffled_index], y_train[shuffled_index]  # shuffled training set

            for fold in range(folds):
                input = x_train[fold * batch:(fold + 1) * batch]
                output = y_train[fold * batch:(fold + 1) * batch]
                neurons, activations = self.feedForward(input, weights_temp)  # calculating current output

                # neural network back propagation
                for level in range(depth - 1, 0, -1):
                    # error calculation
                    if level == depth - 1:  # calculating error for output layer
                        delta = output - neurons[level]
                    else:  # calculating error for hidden layers
                        error = delta.dot(weights_temp[level].T)
                        delta = error * self.hidden_act(activations[level - 1], True) # TODO: Check activations vs function derivative

                    # network update depending on learning technique
                    gradient = neurons[level - 1].T.dot(delta) / batch

                    # update previous network layer with Adam learning
                    vel[level - 1] = b1 * vel[level - 1] + (1 - b1) * gradient  # equivalent to RMS when b1=0
                    cache[level - 1] = b2 * cache[level - 1] + (1 - b2) * gradient ** 2
                    weights_temp[level - 1] += learning_rate * vel[level - 1] / (np.sqrt(cache[level - 1]) + eps)

            # check training and validation progress after each epoch
            self.train_losses.append(self.loss(y_train, self.feedForward(x_train, weights_temp)[0][-1]))
            self.val_losses.append(self.loss(Y_val, self.feedForward(X_val, weights_temp)[0][-1]))
            if self.verbose:
                print("Epoch: %i\tTraining Loss: %.6f\tValidation Loss: %.6f\tLearning Rate: %.0e\tBatch Size: %i" \
                      % (epoch, self.train_losses[-1], self.val_losses[-1], learning_rate, batch))

            # learning rate update
            if epoch > 1:
                if self.val_losses[-1] < min(self.val_losses[:-1]):
                    self.weights = copy.deepcopy(weights_temp)  # Saving best current weights

                improvement = 1 - self.val_losses[-1] / self.val_losses[-2]
                if improvement < tolerance:
                    batch = max(int(batch * sgd_annealing), 1)
                    learning_rate *= rate_annealing
                if epoch > patience and min(self.val_losses[-patience:]) > min(self.val_losses[:-patience]) * (1 - tolerance):
                    if self.verbose: print("Early Stop! Validation did not improve by %f over last %i epochs" %(tolerance,patience))
                    break
            exec_time = time.time() - start_time
            if exec_time >= max_time and max_time > 0:
                if self.verbose: print(
                    "Time limit of %i seconds reached!" % max_time)
                break
        self.epochs += epoch

    # predict output given a certain input to the neural network
    def predict(self, input_value):
        return self.feedForward(input_value, self.weights)[0][-1] # Regressor output

    # to string method
    def __str__(self):
        return "NN_VE%s_LA_%sAC%s_SE%s_EP%s" \
               %(version, ('%d_' * len(self.hidden_layers)) % tuple(self.hidden_layers), self.hidden_act.__name__, self.seed, self.epochs)