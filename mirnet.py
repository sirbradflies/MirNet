"""
    Mix between a Feedforward Neural Network and Restricted Boltzmann Machine.
    Inputs and Outputs are all consolidated and training is a 1-step Gibbs
    sample where the error is the difference between the Input/Output feed
    and their reconstruction after they bounced back (Gibbs' sample)
"""

# TODO: Profile and optimize performance
import time
import copy
import numpy as np
import sklearn.metrics as mt
from sklearn.preprocessing import MinMaxScaler

__version__ = '1.0'
UNCLAMPED_VALUE = 0.0  # DONE: Tested 0 and 0.5


def relu(input_value, minimum=0, maximum=1):
    """
        Apply RELU activation function with option to clip values

        :param input_value: Numpy array with input values
        :param minimum: Minimum value to clip (default 0)
        :param maximum: Maximum value to clip (default 1)
        :return: Numpy array with RELU function applied
    """
    return np.clip(input_value, minimum, maximum)


class MirNet(object):
    """
        Mirror Network that consolidates input and output together
        Training is done similarly to Boltzmann machine with
        a 1-step Gibbs' sampling (deterministic network)
    """

    def __init__(self, hidden_layers=(100,), type='classifier', seed=None,
                 verbose=False):
        """
            Build MirNet basic structure. Loosely structured like Sklean MLP

            :param hidden_layers: Tuple describing the architecture
            and number of neurons present in each layer
            :param type: Network type: 'classifier' (default), 'regressor'
            :param seed: Random seed to initialize the network
            :param verbose: Verbose mode
        """
        if type == "classifier":
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
        self.scaler = MinMaxScaler()  # TESTED: self.scaler = StandardScaler()
        self.verbose = verbose

    def sample(self, input_value, weights):
        """
            Calculate 1-step Gibbs sample of the input data vector

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
            :param weights: Network weights to be used (by default network weights are used)
            :return: Numpy array with the values of the neurons (input/output) calculated
        """
        if weights is None:
            weights = self.weights

        input_neurons = input_array.shape[1]
        total_neurons = weights[0].shape[0]
        samples = len(input_array)
        padding = np.full((samples, total_neurons - input_neurons),
                          UNCLAMPED_VALUE)
        X = self.scaler.transform(np.hstack((input_array, padding)))
        fneurons, bneurons = self.sample(X, weights)

        if input_neurons == total_neurons:
            return self.scaler.inverse_transform(bneurons[0])
        else:
            return self.scaler.inverse_transform(bneurons[0])[:, input_neurons:]  # Return only the fields not passed

    def early_stop(self, epoch, patience, tolerance, start_time, max_time, max_epochs):
        """
            Checks on different training condition to determine whether the
            training should stop

            :param epoch: Current training epoch
            :param patience: Epochs by which is required an improvement of <tolerance> to avoid early stopping
            :param tolerance: Improvement required during <patience> epochs to avoid early stopping
            :param start_time: Time when training started
            :param max_time: Maximum time (in seconds) for training
            :param max_epochs: Maximum number of epochs for training
            :return: Boolean on whether the training should stop
        """
        if epoch > patience:
            best_old_loss = min(self.losses_test[:-patience])
            best_new_loss = min(self.losses_test[-patience:])
            if best_new_loss > best_old_loss * (1 - tolerance):
                print("Early Stop! No %f improvement over last %i epochs"
                      % (tolerance, patience))
                return True

        if max_time > 0 and (time.time() - start_time) >= max_time:
            print("Early Stop! Time limit of %i seconds reached"
                  % max_time)
            return True

        if max_epochs > 0 and epoch >= max_epochs:
            print("Early Stop! Limit of %i epochs reached"
                  % max_epochs)
            return True

        return False

    def fit(self, X, Y=None, sgd_init=100, rate=0.001, m=0.9,
            X_test=None, Y_test=None, test_fraction=0.1, sgd_annealing=0.5,
            tolerance=0.01, patience=10, max_epochs=100, max_time=0):
        """
            Uses a standard SKLearn "fit" interface with Input and Output values and feeds it
            into the train_data method where input and outputs are undifferentiated

            :param X: input values
            :param Y: output or target values (not required)
            :param sgd_init: starting value for mini batch_size size
            :param rate: starting value for learning rate
            :param m: momentum
            :param X_test: Input values for test_data
            :param Y_test: Output values for test_data (not required)
            :param test_fraction: Fraction of X to be used for test_data (if X_test is None)
            :param sgd_annealing: Batch size reduction at each epoch where test_data loss does not improve by tolerance
            :param tolerance: Minimum improvement during <patience> epochs to avoid early stopping
            :param patience: Number of epochs for which is required an improvement of <tolerance> to avoid early stopping
            :param max_epochs: Maximum number of epochs for training
            :param max_time: Maximum time (in seconds) for training
        """
        start_time = time.time()
        data = self.scaler.fit_transform(np.hstack((X, Y)))  # Consolidating input and target batch
        data[np.isnan(data)] = UNCLAMPED_VALUE  # setting default value for unclamped inputs (Nan)
        if Y is None:  # No explicit target, all neurons considered for loss calculation
            targets = data.shape[1]
        else:  # Only target neurons considered for loss calculation
            targets = Y.shape[1]

        samples = len(data)
        shuffled_ix = np.random.permutation(samples)
        if X_test is None:  # Splitting samples in train_data/test_data batches
            test_samples = int(samples * test_fraction)
            test_data = data[shuffled_ix[:test_samples]]
            train_data = data[shuffled_ix[test_samples:]]
        else:
            train_data = data
            test_data = self.scaler.fit_transform(np.hstack((X_test, Y_test)))
            test_samples = len(test_data)

        train_samples = len(train_data)
        layers = (data.shape[1],) + self.hidden_layers
        depth = len(layers)
        if self.weights == []:  # Creates weights if not already present
            for (start, end) in zip(layers[:-1], layers[1:]):
                self.weights.append(np.random.uniform(-0.5, 0.5, (start, end))
                                    * np.sqrt(2 / start))  # HE et al. (2015) initialization
        weights_temp = copy.deepcopy(self.weights)

        batch_size = min(sgd_init, train_samples)  # samples trained at the same time
        weights_update = [np.zeros(weights_temp[d].shape)
                          for d in range(depth - 1)]
        self.losses_train, self.losses_test = [], []
        print("Loss Function: %s\nTraining samples: %i\tTest samples: %i"
              %(self.loss.__name__, train_samples, test_samples))

        for epoch in range(1, max_epochs + 1):  # training the network
            folds = train_samples // batch_size  # iterations per epoch
            for fold in range(folds):
                batch = train_data[fold * batch_size:(fold + 1) * batch_size]
                fwd_values, bkw_values = self.sample(batch, weights_temp)  # calculating current output

                # neural network back propagation
                for layer in range(0, depth - 1):
                    pos_phase = np.dot(np.transpose(fwd_values[layer]),
                                       fwd_values[layer + 1])
                    neg_phase = np.dot(np.transpose(bkw_values[layer]),
                                       bkw_values[layer + 1])
                    delta = rate * (pos_phase - neg_phase) / batch_size
                    weights_update[layer] = m * weights_update[layer] + delta
                    weights_temp[layer] += weights_update[layer]

            # Check training and test_data losses after each epoch
            train_prediction = self.predict(train_data[:, :-targets],
                                            weights=weights_temp)
            self.losses_train.append(self.loss(train_data[:, -targets:],
                                               train_prediction))
            val_prediction = self.predict(test_data[:, :-targets],
                                          weights=weights_temp)
            self.losses_test.append(self.loss(test_data[:, -targets:],
                                              val_prediction))

            print("Epoch: %i\tTraining Loss: %.6f\tValidation Loss: %.6f\tLearning Rate: %.0e\tBatch Size: %i" \
                      % (epoch, self.losses_train[-1], self.losses_test[-1], rate, batch_size))
            self.epochs += 1

            if epoch > 1:
                if self.losses_test[-1] < min(self.losses_test[:-1]):
                    self.weights = copy.deepcopy(weights_temp)  # Saving best current weights
                    batch_size = min(int(batch_size / (1 - sgd_annealing)), sgd_init)
                else:
                    batch_size = max(int(batch_size * (1 - sgd_annealing)), 1)

                if self.early_stop(epoch, patience, tolerance, start_time, max_time, max_epochs):
                    break

    def __str__(self):
        """
        Converts the class to its string representation
        :return: String representing the class
        """
        return "MirNet_VE%s_TY%s_EP%s_LA_%s" \
               % (__version__, self.type, self.epochs,
                  ('%d_' * len(self.hidden_layers)) % tuple(self.hidden_layers))