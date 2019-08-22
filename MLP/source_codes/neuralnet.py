import numpy as np
import sys
from activations import map_of_functions
from derivatives import map_of_derivatives
import downloader

# for the network mentioned in the assignment


class layer(object):
    def __init__(self, no_of_neurons, activation_function='sigmoid'):
        self.size = no_of_neurons

        if activation_function in ["sigmoid", "ReLU", "softmax", "tanh"]:
            self.activation = map_of_functions[activation_function]
            self.derivative = map_of_derivatives[activation_function]

        else:
            print("The activation function mentioned does not exist.")
            sys.exit(0)


class network(object):

    layer_sizes = []
    weights = []
    biases = []
    training_data = []
    test_data = []
    feed_forward_activations = []
    z_values = []
    weight_gradients = []
    bias_gradients = []
    number_of_layers = 0

    def __init__(self, sizes):
        if (len(sizes) > 0):
            self.layer_sizes = sizes
        else:
            print("There is no hidden layer. Not a valid neural network.")
            sys.exit(0)

        for layer_number in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[layer_number]
            fan_out = self.layer_sizes[layer_number+1]
            dl = np.sqrt(6/(fan_in+fan_out))
            self.weights.append(np.random.uniform(-dl, dl, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
        self.number_of_layers = len(self.weights)+1

    def get_data(self):
        self.training_data, self.test_data = downloader.download()

    def cross_entropy_loss(x, y):
        return -np.sum(y*np.log(x))

    def feed_forward(self, x):
        self.feed_forward_activations = []
        temp = x
        for count in range(self.number_of_layers-1):
            z = np.dot(np.transpose(
                self.weights[count]), temp)+self.biases[count]
            self.z_values.append(z)
            a = map_of_functions["softmax"](
                z) if count == self.number_of_layers-2 else map_of_functions["sigmoid"](z)
            self.feed_forward_activations.append(a)
            temp = a
        return self.z_values, self.feed_forward_activations

    def backprop(self, x, y):
        z_vals, a_vals = self.feed_forward(x)
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.biases)
        # for the final layer
        delta = (y/a_vals[-1])*

a = network([784, 500, 250, 100, 10])
a.get_data()
