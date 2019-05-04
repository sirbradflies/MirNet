"""
    Crossover network between a Feedforward Neural Network and Restricted Boltzmann Machine
"""

# TODO: Profile and optimize performance
import time
import copy
import numpy as np
import sklearn.metrics as mt
from sklearn.preprocessing import MinMaxScaler

__version__ = '0.9'
UNCLAMPED_VALUE = 0.0  # DONE: Tested 0 and 0.5


def relu(input_value, min=0, max=1):
    """
        Rectified Linear Unit activation function
        with option to clip values below and above a threshold

        :param input_value: Numpy array with input values
        :param min: Minimum value to clip (default 0)
        :param max: Maximum value to clip (default 1)
        :return: Numpy array
    """
    return np.clip(input_value, min, max)


def logistic(input_value):
    """
        Sigmoid activation function

        :param input_value: Numpy array with input values
        :return: Numpy array
    """
    return 1 / (1 + np.exp(-input_value))


class MirNet(object):
    """
        Definition of the main class
    """

    def __init__(self, hidden_layers=(100,), type='classifier', seed=None, verbose=False):
        """
            Initialization function

            :param hidden_layers: Tuple describing the architecture and number of neurons present in each layer
            :param type: Type of network: 'classifier' (default), 'regressor'
            :param seed: Random seed to initialize the network
            :param verbose: Verbose mode
        """
        if type =="classifier":
            self.loss = mt.log_loss
            self.activation = relu
        elif type == "regressor":
            self.loss = mt.mean_squared_error
            self.activation = relu
        else:
            raise Exception("Type %s not recognized" % type)

        self.type = type
        np.random.seed(seed)
        self.epochs = 0
        self.hidden_layers = hidden_layers
        self.weights = []
        self.scaler = MinMaxScaler() # TESTED: self.scaler = StandardScaler()
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

        # Negative phase, from last to input layer
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
        padding = np.full((samples, total_values - input_values), UNCLAMPED_VALUE)
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
            best_old_solution = min(self.losses_test[:-patience])
            best_current_solution = min(self.losses_test[-patience:])
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

    def fit(self, X, Y=None, sgd_init=100, rate=0.001, m=0.9,
            X_test=None, Y_test=None, test_fraction=0.1, sgd_annealing=0.5,
            tolerance=0.01, patience=10, max_epochs=100, max_time=0):
        """
            Uses a standard SKLearn "fit" interface with Input and Output values and feeds it
            into the train method where input and outputs are undifferentiated

            :param X: input values
            :param Y: output or target values (if present, can also be included in X)
            :param sgd_init: starting value for mini batch_size size
            :param rate: starting value for learning rate
            :param m: momentum
            :param X_test: Input batch for test
            :param Y_test: Output batch for test
            :param test_fraction: Fraction of X to be used for test (if X_test is None)
            :param sgd_annealing: Batch size reduction at each epoch where test loss does not improve by tolerance
            :param tolerance: Minimum improvement during <patience> epochs to avoid early stopping
            :param patience: Number of epochs for which is required an improvement of <tolerance> to avoid early stopping
            :param max_epochs: Maximum number of epochs for training
            :param max_time: Maximum time (in seconds) for training
        """
        start_time = time.time()
        XY = self.scaler.fit_transform(np.hstack((X, Y)))  # Consolidating input and target batch
        XY[np.isnan(XY)] = UNCLAMPED_VALUE  # setting default value for unclamped inputs (Nan)
        if Y is None:  # No explicit target, all neurons considered for loss calculation
            targets = XY.shape[1]
        else:
            targets = Y.shape[1]

        samples = len(XY)
        shuffled_ix = np.random.permutation(samples)
        if X_test is None:
            test_samples = int(samples * test_fraction)
            train = XY[shuffled_ix[test_samples:]]
            test = XY[shuffled_ix[:test_samples]]
        else:
            train = XY
            test = self.scaler.fit_transform(np.hstack((X_test, Y_test)))

        layers = (XY.shape[1],) + self.hidden_layers
        depth = len(layers)
        if self.weights == []:
            for (start, end) in zip(layers[:-1], layers[1:]):  # Start - End start number
                self.weights.append(np.random.uniform(-0.5, 0.5, (start, end))
                                    * np.sqrt(2 / start))  # HE et al. (2015) initialization
        weights_temp = copy.deepcopy(self.weights)

        batch_size = min(sgd_init, len(train))  # samples trained at the same time
        weights_update = [np.zeros(weights_temp[d].shape)
                          for d in range(depth - 1)]
        self.losses_train, self.losses_test = [], []

        for epoch in range(1, max_epochs + 1):  # training the network
            folds = len(train) // batch_size  # iterations per epoch
            for fold in range(folds):
                batch = train[fold * batch_size:(fold + 1) * batch_size]
                fwd_values, bkw_values = self.sample_values(batch, weights_temp)  # calculating current output

                # neural network back propagation
                for layer in range(0, depth - 1):
                    pos_phase = np.dot(np.transpose(fwd_values[layer]),
                                       fwd_values[layer + 1])
                    neg_phase = np.dot(np.transpose(bkw_values[layer]),
                                       bkw_values[layer + 1])
                    delta = rate * (pos_phase - neg_phase) / batch_size
                    weights_update[layer] = m * weights_update[layer] + delta
                    weights_temp[layer] += weights_update[layer]

            # Check training and test losses after each epoch
            train_prediction = self.predict(train[:, :-targets],
                                            weights=weights_temp)
            self.losses_train.append(self.loss(train[:, -targets:],
                                               train_prediction))
            val_prediction = self.predict(test[:, :-targets],
                                          weights=weights_temp)
            self.losses_test.append(self.loss(test[:, -targets:],
                                              val_prediction))

            if self.verbose:
                print("Epoch: %i\tTraining Loss: %.6f\tValidation Loss: %.6f\tLearning Rate: %.0e\tBatch Size: %i" \
                      % (epoch, self.losses_train[-1], self.losses_test[-1], rate, batch_size))

            if epoch > 1:
                if self.losses_test[-1] < min(self.losses_test[:-1]):  # Saving best current weights
                    self.weights = copy.deepcopy(weights_temp)
                if self.early_stop(epoch, patience, tolerance, start_time, max_time, max_epochs):
                    break
                if self.losses_test[-1] > self.losses_test[-2] * (1 - tolerance):  # Improvement too small
                    batch_size = max(int(batch_size * (1 - sgd_annealing)), 1)

        self.epochs += epoch

    def __str__(self):
        """
        Converts the class to its string representation
        :return: String representing the class
        """
        return "MirNet_VE%s_TY%s_EP%s_LA_%s" \
               % (__version__, self.type, self.epochs,
                  ('%d_' * len(self.hidden_layers)) % tuple(self.hidden_layers))