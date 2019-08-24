import numpy as np
import sys
from activations import map_of_functions
from derivatives import map_of_derivatives
import downloader
import random
import matplotlib.pyplot as plt

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
    epochs = 15
    minibatch_losses = []

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
            self.weights.append(np.asarray(
                np.random.uniform(-dl, dl, (fan_in, fan_out)), dtype=np.float64))
            self.biases.append(np.asarray(
                np.zeros((fan_out, 1)), dtype=np.float64))
        self.number_of_layers = len(self.weights)+1

    def get_data(self):
        self.training_data, self.test_data = downloader.download()

    def cross_entropy_loss(self, x, y):
        return -np.sum(y*np.log(x))

    def initialize_gradients(self):
        for weight in self.weights:
            self.weight_gradients.append(np.zeros_like(weight))
        for bias in self.biases:
            self.bias_gradients.append(np.zeros_like(bias))

    def feed_forward(self, x):
        self.feed_forward_activations = []
        zvals = []
        avals = []
        avals.append(x.reshape((-1, 1)))
        temp = x
        for count in range(self.number_of_layers-1):
            z = np.asanyarray(np.dot(np.transpose(
                self.weights[count]), temp)).reshape(self.layer_sizes[count+1], -1)+self.biases[count]
            zvals.append(z)
            a = map_of_functions["softmax"](
                z) if count == self.number_of_layers-2 else map_of_functions["sigmoid"](z)
            avals.append(a)
            temp = a
        return zvals, avals

    def info(self):
        for weight in self.weight_gradients:
            print(np.shape(weight))

    def cross_entropy_derivative_with_softmax(self, x, y):
        return x-y

    def backprop(self, x, y):
        z_vals, a_vals = self.feed_forward(x)
        # self.initialize_gradients()
        delta = self.cross_entropy_derivative_with_softmax(
            a_vals[-1], y)*(map_of_derivatives["softmax"](z_vals[-1]))
        self.bias_gradients[-1] = delta
        self.weight_gradients[-1] = np.dot(a_vals[-2], np.transpose(delta))
        for layer_number in range(2, self.number_of_layers):
            delta = np.multiply(np.dot(self.weights[-layer_number+1], delta),
                                map_of_derivatives["sigmoid"](np.asanyarray(z_vals[-layer_number])))
            self.bias_gradients[-layer_number] = delta
            self.weight_gradients[-layer_number] = np.dot(
                a_vals[-layer_number-1], np.transpose(delta))
        return self.weight_gradients, self.bias_gradients

    def train_network(self, data, learning_rate=0.001, number_of_epochs=15, minibatch_size=64):
        for epoch in range(number_of_epochs):
            total_loss = 0
            print(epoch+1," epoch is running")
            print("--------------------------------")
            random.shuffle(data)
            minibatches = [data[k:k+minibatch_size] for k in range(0,len(data),minibatch_size)]
            for minibatch in minibatches:
                loss = 0
                for _input, one_hot_encoded_vector in minibatch:
                    temp_zvals, temp_avals = self.feed_forward(_input)
                    loss =loss + self.cross_entropy_loss(
                        temp_avals[-1], one_hot_encoded_vector)
                    w_grad, b_grad = self.backprop(
                        _input, one_hot_encoded_vector)
                    for count in range(len(self.weights)):
                        self.weights[count] = self.weights[count] - \
                            learning_rate*w_grad[count]
                        self.biases[count] = self.biases[count] - \
                            learning_rate*b_grad[count]
                # total_loss = total_loss+loss
                loss = loss/minibatch_size
                self.minibatch_losses.append(loss)
                print(f"minibatch loss is ------------ = {loss}")
            total_loss = total_loss/len(data)
            print("****************************************************")

            print(f"Total loss is ------------********** = {total_loss}")

            print("****************************************************")

    def plotter(self):
        plt.plot(self.minibatch_losses)
        plt.show()


a = network([784, 500, 250, 100, 10])
a.get_data()
a.initialize_gradients()
a.train_network(a.training_data)
