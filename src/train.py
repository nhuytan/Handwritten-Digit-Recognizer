import keras
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import load_data
from model import build_model
from preprocessing import preprocess_data


def main():
    batch_size = 128
    epochs = 10
    num_classes = 10

# Load the MNIST dataset using data_loader.py
    x_train, y_train, x_test, y_test = load_data()
    
    # Preprocess the data: reshape, normalize, one-hot encode labels,
    # and create an ImageDataGenerator for data augmentation.

    x_train, y_train, x_test, y_test, datagen = preprocess_data(x_train, y_train, x_test, y_test, num_classes)

    # Print dataset information for verification.
    #print("x_train shape:", x_train.shape)
    #print(x_train.shape[0], "train samples")
    #print(x_test.shape[0], "test samples")




    # Build the CNN model.
    model = build_model(input_shape=(28, 28, 1), num_classes=num_classes)
    
    # Compile the model using the Adam optimizer and categorical crossentropy loss.
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Train the model using the data generator for on-the-fly data augmentation.
    hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                     epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    #print("The model has successfully trained")

    # Save the trained model to a file.
    model.save('mnist_v02_50epoch.h5')
    #print("Saving the model as mnist_v02_50epoch.h5")


if __name__ == '__main__':
    main()