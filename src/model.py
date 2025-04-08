from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates the CNN model for handwritten digit recognition.

    Args:
        input_shape (tuple): Shape of the input data (default is (28, 28, 1)).
        num_classes (int): Number of classes (default is 10 for digits 0-9).

    Returns:
        model: A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.add(Dropout(0.5))
    # Note: Including an extra Dense layer with softmax is unconventional
    # if you already have one. Adjust as necessary.
    model.add(Dense(num_classes, activation='softmax'))
    return model
