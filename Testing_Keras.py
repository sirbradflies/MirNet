import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from mnist import MNIST
import tensorflow as tf
import itertools


def numerai_dataset():
    training_data = pd.read_csv('datasets/numerai_training_data.csv', header=0)
    tournament_data = pd.read_csv('datasets/numerai_tournament_data.csv')
    features = [f for f in list(training_data) if "feature" in f]
    tournament_data = tournament_data[tournament_data["data_type"] == "validation"]
    X = training_data[features].values
    Y = training_data["target_bernie"].values.reshape((X.shape[0], 1))
    X_val = tournament_data[features].values
    Y_val = tournament_data["target_bernie"].values.reshape((X_val.shape[0], 1))
    out_activation = "sigmoid"
    loss_function = "binary_crossentropy"
    metrics = ['binary_crossentropy']
    return X, Y, X_val, Y_val, out_activation, loss_function, metrics


def mnist_dataset():
    mndata = MNIST('datasets/')
    X, Y_labels = mndata.load_training()
    X = np.asanyarray(X) / 255
    Y_labels = np.asanyarray(Y_labels)
    X_val, Y_val_labels = mndata.load_testing()
    X_val = np.asanyarray(X_val) / 255
    Y_val_labels = np.asanyarray(Y_val_labels)
    out_activation = "sigmoid"
    loss_function = "sparse_categorical_crossentropy"
    metrics = ['accuracy']
    return X, Y_labels, X_val, Y_val_labels, out_activation, loss_function, metrics


def main():
    # Training neural network
    param_name = "Dropout"
    param = list(itertools.product([0,0.1,0.25,0.5],
                                   ["relu"],
                                   ["nmr"],
                                   [100]))

    for p in param:
        # Setting up model parameters
        depth = 3
        activation = p[1]
        dataset = p[2]
        epochs = 100  # number of iterations to exit the training loop
        tests = 5
        neurons = p[3]
        if dataset == "nmr":
            X, Y, X_val, Y_val, out_activation, loss, metrics = numerai_dataset()
        else:
            X, Y, X_val, Y_val, out_activation, loss, metrics = mnist_dataset()

        print("%s: %s" % (param_name, p))
        runTime = 0
        loss_val = 0
        losses_mx = [0] * epochs
        for seed in range(tests):  # random number generator seed
            print("Seed: %s" % seed)
            start = time.clock()
            # Model training
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(neurons,
                                            activation=activation,
                                            input_shape=X.shape[1:]))
            model.add(tf.keras.layers.Dropout(p[0]))
            for d in range(1,depth):
                model.add(tf.keras.layers.Dense(neurons,
                                                activation=activation))

            if len(Y.shape) < 2:
                outputs = len(np.unique(Y))
            else:
                outputs = Y.shape[1]
            model.add(tf.keras.layers.Dense(outputs, activation=out_activation))

            model.compile(optimizer='adam', loss=loss, metrics=metrics)
            print("%s %s Testing - %s" % (param_name, dataset, model))
            calls = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                        min_delta=0,
                                                        patience=3,
                                                        mode='auto',
                                                        restore_best_weights=True)]
            losses = model.fit(X, Y, validation_data=(X_val,Y_val),
                               epochs=epochs, verbose=2,
                               callbacks=calls).history[metrics[0]]
            runTime += (time.clock() - start) / tests
            losses_mx = np.add(losses_mx, np.pad(losses,(0,epochs-len(losses)),"edge"))
            loss_val += model.evaluate(X_val, Y_val)[1] / tests

        # plot results
        # plt.plot(losses.history["loss"],label="PA%s_TI%i" %(p,runTime))
        for mx in metrics:
            plt.plot(losses_mx, label="PA%s_TI%i_VALLO%f" % (p, runTime, loss_val))

    plt.title("%s %s Testing" % (param_name, dataset))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
