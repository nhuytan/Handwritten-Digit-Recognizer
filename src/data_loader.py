
from keras.datasets import mnist

def load_data():
    """
    Loads the MNIST dataset.

    Returns:
        x_train, y_train, x_test, y_test: Arrays containing the data and labels.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test
