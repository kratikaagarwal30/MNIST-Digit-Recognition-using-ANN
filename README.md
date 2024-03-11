# MNIST-Digit-Recognition-using-ANN

Project Overview
This project focuses on building and training an Artificial Neural Network (ANN) to recognize handwritten digits from the MNIST dataset. The goal is to accurately classify images of handwritten digits (0-9) using deep learning techniques.

Dataset:
The MNIST dataset comprises 70,000 images of handwritten digits, each of which is 28x28 pixels. It is split into a training set of 60,000 images and a test set of 10,000 images. This dataset is widely used for training and testing in the field of machine learning.

Prerequisites:
TensorFlow
Keras
NumPy
Matplotlib

Model Architecture
The model architecture consists of:

An input layer that flattens the 28x28 images into a 784-dimensional vector.
Several dense layers with ReLU activation functions for feature extraction and non-linearity.
A final dense layer with a softmax activation function to output the probability distribution across the 10 digit classes.

Results
After training, the model achieved an accuracy of 97% on the MNIST test set. Further details and analysis of the model's performance, including confusion matrices or incorrect predictions, are provided within the project notebook.

Conclusion
This project highlights the effectiveness of simple ANN architectures in classifying images, such as handwritten digits from the MNIST dataset. Future work could explore more complex models, data augmentation techniques, or hyperparameter tuning to further improve performance.
