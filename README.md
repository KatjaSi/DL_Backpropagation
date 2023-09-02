# DL_Backpropagation
Implementing Backpropagation algorithm in DL

Backpropagation is a widely used algorithm for training neural networks. Backpropagation is a form of gradient descent that allows the network to learn from examples and adjust its weights and biases to better fit the training data. This package provides a simple implementation of backpropagation that can be easily extended and customized.

# Usage
To use DL_Backpropagation, first clone or download the repository from GitHub:


git clone https://github.com/KatjaSi/DL_Backpropagation.git

# Implementation
The backpropagation is implemented using a simple object-oriented approach. The Network class represents a neural network, and it consists of a list of Layer objects, each of which consists of a list of Neuron objects. The Layer class is responsible for computing the output of its neurons given the inputs from the previous layer, and the Neuron class is responsible for computing its output given its inputs and its weights and biases. The package also provides several activation functions that can be used with the neurons, including the sigmoid function and the rectified linear unit (ReLU) function.

The Network class provides a train method that takes a set of input-output pairs, a number of epochs, and a learning rate. During training, the network computes the output for each input using the current weights and biases, compares it to the target output, and adjusts the weights and biases to reduce the error. This is done using the backpropagation algorithm, which computes the error at each neuron and propagates it backwards through the network to adjust the weights and biases. The train method returns the final training error.

The Network class also provides a predict method that takes an input and returns the output of the network for that input. This can be used to test the network after training.
