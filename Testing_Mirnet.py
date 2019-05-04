import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn import metrics as mt
from mnist import MNIST
import mirnet as mn
import itertools


def numerai_dataset():
    training_data = pd.read_csv('datasets/numerai_training_data.csv', header=0)
    tournament_data = pd.read_csv('datasets/numerai_tournament_data.csv')
    features = [f for f in list(training_data) if "feature" in f]
    tournament_data = tournament_data[tournament_data["data_type"] == "validation"]
    scaler = preprocessing.MinMaxScaler()
    X = training_data[features].values
    Y = training_data["target_bernie"].values.reshape((X.shape[0], 1))
    Y = scaler.fit_transform(Y)
    X_val = tournament_data[features].values
    Y_val = tournament_data["target_bernie"].values.reshape((X_val.shape[0], 1))
    Y_val = scaler.transform(Y_val)
    return X, Y, X_val, Y_val, mt.log_loss, scaler


def mnist_dataset():
    mndata = MNIST('datasets/')
    X, Y_labels = mndata.load_training()
    X = np.asanyarray(X) / 255
    Y_labels = np.reshape(np.asanyarray(Y_labels), (len(Y_labels), 1))
    binarizer = preprocessing.LabelBinarizer()
    Y = binarizer.fit_transform(Y_labels)
    X_val, Y_val_labels = mndata.load_testing()
    X_val = np.asanyarray(X_val) / 255
    Y_val_labels = np.reshape(np.asanyarray(Y_val_labels), (len(Y_val_labels), 1))
    Y_val = binarizer.fit_transform(Y_val_labels)
    loss_function = mt.accuracy_score
    return X[:1000], Y[:1000], X_val[:1000], Y_val[:1000], loss_function, binarizer


def main():
    # Training neural network
    param_name = "SGD Annealing"
    #param = list(itertools.product([1],
    #                               [mn.MirNet],
    #                               ["nmr"],
    #                               [50]))
    #param = [0.5,0.9,0.99,1.0]
    param = [1.0]

    for p in param:
        network = mn.MirNet
        dataset = "mnist"
        if dataset == "nmr":
            X, Y, X_val, Y_val, loss_function, transformer = numerai_dataset()
        else:
            X, Y, X_val, Y_val, loss_function, transformer = mnist_dataset()

        # Setting up model parameters
        epochs = 1000  # number of iterations to exit the training loop
        tolerance = 10**-3
        tests = 1
        rate_init = 10**-4
        patience = 100
        max_time = 60 * 60 * 1
        depth = 1
        type = "classifier"
        neurons = 500
        sgd_annealing = p
        layers = (neurons,) * depth

        print("%s: %s" % (param_name,p))
        train_loss,val_loss = np.zeros(epochs),np.zeros(epochs)
        best_train_loss, best_val_loss = 0,0
        runTime = 0
        for seed in range(tests):  # random number generator seed
            print("Seed: %s" % seed)
            start = time.clock()

            # Model training
            net = network(hidden_layers=layers, type=type, seed=seed, verbose=True)
            print("%s %s Testing - %s" % (param_name, dataset, net))
            sgd = int(X.shape[0] / 10)
            net.fit(X, Y, sgd_init=sgd, tolerance=tolerance, max_epochs=epochs, max_time=max_time,
                    patience=patience, rate=rate_init, X_test=X_val, Y_test=Y_val,
                    sgd_annealing=sgd_annealing)
            runTime += (time.clock()-start)/tests

            train_loss += np.pad(net.losses_train, (0, epochs - len(net.losses_train)), "edge") / tests
            Y_labels = transformer.inverse_transform(Y)
            train_prediction = transformer.inverse_transform(net.predict(X))
            best_train_loss += loss_function(Y_labels,train_prediction) / tests

            val_loss += np.pad(net.losses_test, (0, epochs - len(net.losses_test)), "edge") / tests
            Y_val_labels = transformer.inverse_transform(Y_val)
            val_prediction = transformer.inverse_transform(net.predict(X_val))
            best_val_loss += loss_function(Y_val_labels,val_prediction) / tests

        # plot results
        #plt.plot(train_loss,label="PA%s_TI%i_TRALO%.5f" %(p,runTime,best_train_loss))
        plt.plot(val_loss, label="PA%s_TI%i_VALLO%.5f" % (p, runTime, best_val_loss))
        #pd.DataFrame(net.weights[0]).to_csv("weights.csv")

    x = print(net)
    plt.title("%s %s Testing" % (param_name, dataset))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()