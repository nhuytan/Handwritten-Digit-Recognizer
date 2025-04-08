# Handwritten Digit Recognizer

This project implements a handwritten digit recognizer using a convolutional neural network (CNN) and the MNIST dataset. It demonstrates data preprocessing, model training with data augmentation, model evaluation, and a GUI for recognizing drawn digits.

## Project Structure

Handwritten-Digit-Recognizer
├── README.md
├── src
│   ├── **init**.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── gui.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
├── setup.py
├── requirement.txt
└── notebook
└── Handwritten_Digit_Recognizer.ipynb

### Create a Python environment:

- For Python3 users:
  - `pip install virtualenv`
  - `virtualenv venv_name`
  - `source path/to/venv_name activate`
- For Anaconda users:
  - `conda create --name conda_env`
  - `conda activate conda_env`
  - `conda install pip`

### Install required library

- `pip install -r requirement.txt`

### USAGE

- Train model : python src/train.py
- Launch GUI : python src/gui.py (Press key 'n' to clear the screen at any time and draw a new digit.)
