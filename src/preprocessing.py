import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(x_train, y_train, x_test, y_test, num_classes=10):
    """
    Preprocesses the MNIST image data and labels.
    
    The function reshapes the images to add a channel dimension, converts the class labels
    to one-hot encoding, casts the image data to float32, normalizes the pixel values to the range [0, 1],
    and creates an ImageDataGenerator for data augmentation.

    Args:
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Testing images.
        y_test (numpy.ndarray): Testing labels.
        num_classes (int, optional): Number of classes. Defaults to 10.
        
    Returns:
        tuple: preprocessed x_train, y_train, x_test, y_test, and the ImageDataGenerator instance.
    """
    # Reshape images to add channel dimension (assumes MNIST images with shape 28x28)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Convert class labels to one-hot encoded vectors
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Convert image datatype to float32 and normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,       # Rotate images by up to 10 degrees
        zoom_range=0.1,          # Randomly zoom images by up to 10%
        width_shift_range=0.1,   # Shift images horizontally by up to 10% of total width
        height_shift_range=0.1,  # Shift images vertically by up to 10% of total height
        horizontal_flip=False,   # Do not flip images horizontally
        vertical_flip=False      # Do not flip images vertically
    )

    # Fit the data generator on training images (required for some augmentation types)
    datagen.fit(x_train)

    return x_train, y_train, x_test, y_test, datagen
