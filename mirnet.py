"""
    Crossover network between a Feedforward Neural Network and Restricted Boltzmann Machine
"""

# TODO: Profile and optimize performance

import time
import copy
import numpy as np
import sklearn.metrics as mt
from sklearn.preprocessing import MinMaxScaler

__version__ = '0.7'
UNCLAMPED_VALUE_DEFAULT = 0.0  # DONE: Tested 0 and 0.5


def relu(input_value, min=0, max=1):
    """
        Rectified Linear Unit activation function
        with option to clip values below and above a threshold

        :param input: Numpy array with input values
        :param min: Minimum value to clip (default 0)
        :param max: Maximum value to clip (default 1)
        :return: Numpy array
    """
    return np.clip(input_value, min, max)


def logistic(input_value):
    """
        Sigmoid activation function

        :param input: Numpy array with input values
        :return: Numpy array
    """
    return 1 / (1 + np.exp(-input_value))


def expand_network(weights_curr, layer, neurons_new=10, verbose=False):
    if verbose:
        print("Expanding layer %i with %i neurons" % (layer, neurons_new))
    weights_new = copy.deepcopy(weights_curr)

    if layer > 0:
        # Not 1st layer, adding weights before new neurons
        inbound = weights_new[layer-1].shape[0]
        neurons_total = weights_new[layer-1].shape[1] + neurons_new
        weights_add = np.random.uniform(-0.5, +0.5, (inbound, neurons_new)) \
                      / np.sqrt(2 / neurons_total)
        weights_new[layer-1] = np.hstack((weights_new[layer-1],weights_add))

    if layer < len(weights_curr)-1:
        # Not last layer, adding weights after new neurons
        outbound = weights_new[layer].shape[1]
        neurons_total = weights_new[layer].shape[0] + neurons_new
        weights_add = np.random.uniform(-0.5, +0.5, (neurons_new, outbound)) \
                      / np.sqrt(2 / neurons_total)
        weights_new[layer] = np.hstack((weights_new[layer], weights_add))

    return weights_new


class MirNet(object):
    """
        Definition of the main class
    """

    def __init__(self, hidden_layers=(100,), activation="relu", seed=None, verbose=False):
        """
            Initialization function

            :param hidden_layers: Tuple describing the architecture and number of neurons present in each layer
            :param activation: Activation function ('relu', 'logistic')
            :param seed: Random seed to initialize the network
            :param verbose: Verbose mode
        """
        self.seed = seed
        np.random.seed(self.seed)
        self.epochs = 0
        self.loss = mt.mean_squared_error
        self.hidden_layers = hidden_layers
        self.activation = globals().copy()[activation]
        self.weights = []
        self.scaler = MinMaxScaler() # DONE: self.scaler = StandardScaler()
        self.verbose = verbose


    def sample_values(self, input_value, weights):
        """
            Return a 1-step Gibbs sample of the input data vector

            :param input_value: Numpy array with values for all first level neurons (including output)
            :param weights: List of Numpy arrays with network weights
            :return: Two Numpy arrays with neurons value calculated for the positive and negative phase
        """
        # Positive phase, from input to last layer
        pos_phase = [input_value]
        for w in weights:
            neurons_input = np.dot(pos_phase[-1], w)
            neurons_output = self.activation(neurons_input)
            pos_phase = pos_phase + [neurons_output]

        # Negative phase, from last layer to input
        neg_phase = [pos_phase[-1]]
        for w in weights[::-1]:
            neurons_input = np.dot(neg_phase[0], np.transpose(w))
            neurons_output = self.activation(neurons_input)
            neg_phase = [neurons_output] + neg_phase

        return pos_phase, neg_phase


    def predict(self, input_array, weights=None):
        """
            Predict output given a certain input to the network.
            If not all columns are passed (values "unclamped") only missing fields are returned

            :param input_array: Numpy array with values for first level neurons
            :param weights: Network weights to be used (default are the network weights)
            :return: Numpy array with the values of the neurons (input/output) calculated
        """
        if weights is None:
            weights = self.weights

        input_values = input_array.shape[1]
        total_values = weights[0].shape[0]
        samples = len(input_array)
        padding = np.full((samples, total_values - input_values), UNCLAMPED_VALUE_DEFAULT)
        X = self.scaler.transform(np.hstack((input_array, padding)))
        fneurons, bneurons = self.sample_values(X, weights)

        if input_values == total_values:
            return self.scaler.inverse_transform(bneurons[0])
        else:
            return self.scaler.inverse_transform(bneurons[0])[:, input_values:]  # Return only the fields not passed


    def early_stop(self, epoch, patience, tolerance, start_time, max_time, max_epochs):
        """
            Checks on different training condition to determine whether the
            training should stop

            :param epoch: Current training epoch
            :param patience: Number of epochs for which is required an improvement of <tolerance> to avoid early stopping
            :param tolerance: Minimum improvement during <patience> epochs to avoid early stopping
            :param start_time: Time when training started
            :param max_time: Maximum time (in seconds) for training
            :param max_epochs: Maximum number of epochs for training
            :return: Boolean on whether the training should stop
        """
        if epoch > patience:
            best_old_solution = min(self.val_losses[:-patience])
            best_current_solution = min(self.val_losses[-patience:])
            if best_current_solution > best_old_solution * (1 - tolerance):
                if self.verbose: print(
                    "Early Stop! Validation did not improve by %f over last %i epochs"
                    % (tolerance, patience))
                return True

        if max_time > 0 and (time.time() - start_time) >= max_time:
            if self.verbose: print(
                "Time limit of %i seconds reached!" % max_time)
            return True

        if max_epochs > 0 and epoch >= max_epochs:
            if self.verbose: print(
                "Limit of %i epochs reached!" % max_epochs)
            return True

        return False


    def fit(self, X, Y, sgd_init=100, rate_init=0.001, m=0.9,
            X_val=None, Y_val=None, val_fraction=0.1, sgd_annealing=0.5,
            tolerance=0.01, patience=10, max_epochs=100, max_time=0):
        """
            Uses a standard SKLearn "fit" interface with Input and Output values and feeds it
            into the train method where input and outputs are undifferentiated

            :param X: input values
            :param Y: output values (targets)
            :param sgd_init: starting value for mini batch_size size
            :param rate_init: starting value for learning rate
            :param m: momentum
            :param X_val: Input data for validation
            :param Y_val: Output data for validation
            :param val_fraction: Fraction of X to be used for validation (if X_validation is None)
            :param sgd_annealing: Batch size reduction at each epoch where validation loss does not improve by tolerance
            :param tolerance: Minimum improvement during <patience> epochs to avoid early stopping
            :param patience: Number of epochs for which is required an improvement of <tolerance> to avoid early stopping
            :param max_epochs: Maximum number of epochs for training
            :param max_time: Maximum time (in seconds) for training
        """
        X_cons = np.hstack((X, Y))  # Consolidating input and target data
        tgt_neurons = Y.shape[1]
        if X_val is not None and Y_val is not None:
            X_val_cons = np.hstack((X_val, Y_val))
        else:
            X_val_cons = None
        self.fit_all(X_cons, sgd_init, rate_init, m, X_val_cons,
                     val_fraction, tgt_neurons, sgd_annealing,
                     tolerance, patience, max_epochs, max_time)

    def fit_all(self, X, sgd_init=100, rate_init=0.001, m=0.9, X_val=None,
                val_fraction=0.1, tgt_neurons=None, sgd_annealing=0.5,
                tolerance=0.01, patience=10, max_epochs=100, max_time=0):
        """
            Train the network through a mix of CD 0.5 and Hebbian learning

            :param X: data values (no distinction between "data" and "output" or "target" values)
            :param sgd_init: starting value for mini batch_size size
            :param rate_init: starting value for learning rate
            :param m: momentum
            :param X_val: Input data for validation (no distinction between "data" and "output" or "target" values)
            :param val_fraction: Fraction of X to be used for validation
            :param tgt_neurons: Neurons to be considered for target loss calculation
            :param sgd_annealing: Batch size reduction at each epoch where validation loss does not improve by <tolerance>
            :param tolerance: Minimum improvement during <patience> epochs to avoid early stopping
            :param patience: Number of epochs for which is required an improvement of <tolerance> to avoid early stopping
            :param max_epochs: Maximum number of epochs for training
            :param max_time: Maximum time (in seconds) for training
        """
        start_time = time.time()
        nan_mask = np.isnan(X)
        X_std = self.scaler.fit_transform(X)

        layers = (X_std.shape[1],) + self.hidden_layers
        depth = len(layers)
        #  TODO: create dedicated function with also add feature
        if self.weights == []:
            for (start, end) in zip(layers[:-1], layers[1:]):  # Start - End start number
                self.weights.append(np.random.uniform(-0.5, 0.5, (start, end))
                                    * np.sqrt(2 / start))  # HE et al. (2015) initialization
        weights_temp = copy.deepcopy(self.weights)

        samples = len(X_std)
        shuffled_ix = np.random.permutation(samples)
        if X_val is None:
            val_samples = int(samples * val_fraction)
            X_val_std = X_std[shuffled_ix[:val_samples]]
            X_train = X_std[shuffled_ix[val_samples:]]
        else:
            X_val_std = self.scaler.fit_transform(X_val)
            X_train = X_std

        if tgt_neurons is None: # TODO: Validate and Engineer Validation Neurons use
            tgt_neurons = X_val_std.shape[1]

        if self.verbose:
            print("Target Neurons for validation loss: %i" % tgt_neurons)

        batch_size = min(sgd_init, len(X_train))  # samples trained at the same time
        rate = rate_init
        network_inc = [np.zeros(weights_temp[d].shape) for d in range(depth-1)]
        self.train_losses, self.val_losses = [], []

        for epoch in range(1, max_epochs + 1):  # training the network
            X_std[nan_mask] = self.predict(X_std)[nan_mask]  # setting default value for unclamped inputs (Nan)
            folds = len(X_train) // batch_size  # iterations per epoch

            for fold in range(folds):
                data = X_train[fold * batch_size:(fold + 1) * batch_size]
                fwd_neurons, bkw_neurons = self.sample_values(data, weights_temp)  # calculating current output

                # neural network back propagation
                for level in range(0, depth - 1):
                    pos_phase = np.dot(np.transpose(fwd_neurons[level]), fwd_neurons[level + 1])
                    neg_phase = np.dot(np.transpose(bkw_neurons[level]), bkw_neurons[level + 1])
                    update_matrix = rate * (pos_phase - neg_phase) / batch_size
                    network_inc[level] = m * network_inc[level] + update_matrix
                    weights_temp[level] += network_inc[level]

            # check training and validation progress after each epoch
            train_prediction = self.predict(X_train, weights=weights_temp)
            self.train_losses.append(self.loss(X_train, train_prediction))
            # Validation loss is calculated
            val_prediction = self.predict(X_val_std[:,:-tgt_neurons], weights=weights_temp)
            self.val_losses.append(
                self.loss(X_val_std[:, -tgt_neurons:], val_prediction)
            )

            if self.verbose:
                print("Epoch: %i\tTraining Loss: %.6f\tValidation Loss: %.6f\tLearning Rate: %.0e\tBatch Size: %i" \
                      % (epoch, self.train_losses[-1], self.val_losses[-1], rate, batch_size))

            if epoch > 1:
                if self.val_losses[-1] < min(self.val_losses[:-1]):  # Saving best current weights
                    self.weights = copy.deepcopy(weights_temp)
                    batch_size = min(int(batch_size / np.sqrt(1 - sgd_annealing)), sgd_init)
                    #expand_layer = 1
                else:
                    batch_size = max(int(batch_size * (1 - sgd_annealing)),1)
                if self.early_stop(epoch, patience, tolerance, start_time, max_time, max_epochs):
                    break
                """
                if expand_layer > len(weights_temp):
                    neurons = 1
                else:
                    neurons = weights_temp[expand_layer-1].shape[1]
                weights_temp = expand_network(weights_temp, expand_layer, neurons, verbose=self.verbose)
                network_inc = [np.zeros(weights_temp[d].shape) for d in range(depth-1)]
                expand_layer += 1
                """
        self.epochs += epoch


    def __str__(self):
        """
        Converts the class to its string representation
        :return: String representing the class
        """
        return "MirNet_VE%s_AC%s_LA_%s_SE%s_EP%s" \
               % (__version__, self.activation.__name__,
                  ('%d_' * len(self.hidden_layers)) % tuple(self.hidden_layers),
                  self.seed, self.epochs)