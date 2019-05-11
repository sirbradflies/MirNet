import numpy as np
import mirnet as mn
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics as mt


def main():
    mndata = MNIST('datasets/')
    # Preparing training data
    X, Y_labels = mndata.load_training()
    X = np.asanyarray(X) / 255
    Y_labels = np.reshape(np.asanyarray(Y_labels), (len(Y_labels), 1))
    binarizer = preprocessing.LabelBinarizer()
    Y = binarizer.fit_transform(Y_labels)
    # Preparing test data
    X_test, Y_test_labels = mndata.load_testing()
    X_test = np.asanyarray(X_test) / 255
    Y_test_labels = np.reshape(np.asanyarray(Y_test_labels), (len(Y_test_labels), 1))
    Y_test = binarizer.fit_transform(Y_test_labels)

    # Setting up model parameters
    depth = 1
    neurons = 100
    epochs = 100
    patience = 5
    sgd = int(len(X)/10)
    layers = (neurons,) * depth

    # Training the network
    net = mn.MirNet(hidden_layers=layers, type="classifier", verbose=True)
    net.fit(X, Y, sgd_init=sgd, max_epochs=epochs, patience=patience,
            X_test=X_test, Y_test=Y_test)

    Y_labels = binarizer.inverse_transform(Y)
    train_prediction = binarizer.inverse_transform(net.predict(X))
    best_train_loss = mt.accuracy_score(Y_labels,train_prediction)
    test_prediction = binarizer.inverse_transform(net.predict(X_test))
    best_test_loss = mt.accuracy_score(Y_test_labels, test_prediction)

    # Plotting losses
    plt.title("MirNet MNIST Training")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(net.losses_train,label="TRAIN ACCURACY %.4f" % best_train_loss)
    plt.plot(net.losses_test, label="TEST ACCURACY%.4f" % best_test_loss)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()