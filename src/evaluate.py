import keras
from keras.models import load_model
from keras.utils import to_categorical
from src.data_loader import load_data

def preprocess_data(x, y, num_classes=10):
    """
    Reshapes and normalizes image data and converts labels to one-hot encoding.
    """
    x = x.reshape(x.shape[0], 28, 28, 1).astype('float32') / 255.0
    y = to_categorical(y, num_classes)
    return x, y

def main():
    # Load test data
    _, _, x_test, y_test = load_data()
    x_test, y_test = preprocess_data(x_test, y_test)
    
    # Load the trained model (ensure the correct filename is used)
    model = load_model('mnist_v02_50epoch.h5')
    
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()
