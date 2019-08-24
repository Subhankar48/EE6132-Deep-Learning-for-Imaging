import numpy as np
import sys
from activations import map_of_functions
from derivatives import map_of_derivatives
import downloader
import random
import matplotlib.pyplot as plt
import evaluations as ev

# for the network mentioned in the assignment


class layer(object):
    def __init__(self, no_of_neurons, activation_function='sigmoid'):
        self.size = no_of_neurons

        if activation_function in ["sigmoid", "tanh", "softmax", "tanh"]:
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
    minibatch_losses = []
    _accuracy = 0
    _precision = 0
    _recall = 0
    _f1_score = 0

    def __init__(self, sizes):
        if (len(sizes) > 0):
            self.layer_sizes = sizes
        else:
            print("There is no hidden layer. Not a valid neural network.")
            sys.exit(0)

        for layer_number in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[layer_number]
            fan_out = self.layer_sizes[layer_number+1]
            self.weights.append(self.glorot_initialization(fan_in, fan_out))
            self.biases.append(np.asarray(
                np.zeros((fan_out, 1)), dtype=np.float64))
        self.number_of_layers = len(self.weights)+1

    def get_data(self):
        self.training_data, self.test_data = downloader.download()

    def glorot_initialization(self, fan_in, fan_out):
        dl = np.sqrt(6/(fan_in+fan_out))
        return np.asarray(np.random.uniform(-dl, dl, (fan_in, fan_out)), dtype=np.float64)

    def cross_entropy_loss(self, x, y):
        return -(y*np.log(x)).mean()

    def initialize_gradients(self):
        for weight in self.weights:
            self.weight_gradients.append(np.zeros_like(weight))
        for bias in self.biases:
            self.bias_gradients.append(np.zeros_like(bias))

    # def feed_forward(self, x):
    #     self.feed_forward_activations = []
    #     zvals = []
    #     avals = []
    #     avals.append(x.reshape((-1, 1)))
    #     temp = x
    #     for count in range(self.number_of_layers-1):
    #         z = np.asanyarray(np.dot(np.transpose(
    #             self.weights[count]), temp)).reshape(self.layer_sizes[count+1], -1)+self.biases[count]
    #         zvals.append(z)
    #         a = map_of_functions["softmax"](
    #             z) if count == self.number_of_layers-2 else map_of_functions["sigmoid"](z)
    #         avals.append(a)
    #         temp = a
    #     return zvals, avals

    def feed_forward(self, x):
        self.feed_forward_activations = []
        zvals = []
        avals = []
        avals.append(x)
        temp = x
        for count in range(self.number_of_layers-1):
            z = np.asanyarray(np.dot(np.transpose(
                self.weights[count]), temp))+self.biases[count]
            zvals.append(z)
            a = map_of_functions["softmax"](
                z) if count == self.number_of_layers-2 else map_of_functions["tanh"](z)
            avals.append(a)
            temp = a
        return zvals, avals

    def cross_entropy_derivative_with_softmax(self, x, y):
        return x-y

    # def backprop(self, x, y):
    #     z_vals, a_vals = self.feed_forward(x)
    #     # self.initialize_gradients()
    #     delta = self.cross_entropy_derivative_with_softmax(
    #         a_vals[-1], y)*(map_of_derivatives["softmax"](z_vals[-1]))
    #     self.bias_gradients[-1] = delta
    #     self.weight_gradients[-1] = np.dot(a_vals[-2], np.transpose(delta))
    #     for layer_number in range(2, self.number_of_layers):
    #         delta = np.multiply(np.dot(self.weights[-layer_number+1], delta),
    #                             map_of_derivatives["sigmoid"](np.asanyarray(z_vals[-layer_number])))
    #         self.bias_gradients[-layer_number] = delta
    #         self.weight_gradients[-layer_number] = np.dot(
    #             a_vals[-layer_number-1], np.transpose(delta))
    #     return self.weight_gradients, self.bias_gradients

    def backprop(self, x, y):
        z_vals, a_vals = self.feed_forward(x)
        # self.initialize_gradients()
        delta = self.cross_entropy_derivative_with_softmax(
            a_vals[-1], y)*(map_of_derivatives["softmax"](z_vals[-1]))
        self.bias_gradients[-1] = np.mean(delta, axis=1).reshape(-1, 1)
        self.weight_gradients[-1] = np.dot(a_vals[-2], np.transpose(delta))
        for layer_number in range(2, self.number_of_layers):
            delta = np.multiply(np.dot(self.weights[-layer_number+1], delta),
                                map_of_derivatives["tanh"](np.asanyarray(z_vals[-layer_number])))
            self.bias_gradients[-layer_number] = np.mean(
                delta, axis=1).reshape(-1, 1)
            self.weight_gradients[-layer_number] = np.dot(
                a_vals[-layer_number-1], np.transpose(delta))
        return self.weight_gradients, self.bias_gradients

    # def train_network(self, data, learning_rate=0.01, number_of_epochs=15, minibatch_size=64):
    #     for epoch in range(number_of_epochs):
    #         total_loss = 0
    #         print(epoch+1, " epoch is running")
    #         print("--------------------------------")
    #         random.shuffle(data)
    #         minibatches = [data[k:k+minibatch_size]
    #                        for k in range(0, len(data), minibatch_size)]
    #         for minibatch in minibatches:
    #             loss = 0
    #             for _input, one_hot_encoded_vector in minibatch:
    #                 temp_zvals, temp_avals = self.feed_forward(_input)
    #                 loss = loss + self.cross_entropy_loss(
    #                     temp_avals[-1], one_hot_encoded_vector)
    #                 w_grad, b_grad = self.backprop(
    #                     _input, one_hot_encoded_vector)
    #                 for count in range(len(self.weights)):
    #                     self.weights[count] = self.weights[count] - \
    #                         learning_rate*w_grad[count]/minibatch_size
    #                     self.biases[count] = self.biases[count] - \
    #                         learning_rate*b_grad[count]/minibatch_size
    #             total_loss = total_loss+loss
    #             loss = loss/minibatch_size
    #             self.minibatch_losses.append(loss)
    #             print(f"minibatch loss is ------------ = {loss}")
    #         total_loss = total_loss/len(data)
    #         print("****************************************************")

    #         print(f"Total loss is ------------********** = {total_loss}")

    #         print("****************************************************")

    def train_network(self, data, learning_rate=0.01, number_of_epochs=1, minibatch_size=64):
        pixel_values = data[0]
        labels = data[1]
        temp_list = list(zip(pixel_values, labels))    
        for epoch in range(number_of_epochs):
            print(epoch+1, " epoch is running")
            print("--------------------------------")
            random.shuffle(temp_list)
            pixel_values, labels = zip(*temp_list)

            input_batches = [pixel_values[k:k+minibatch_size]
                             for k in range(0, len(pixel_values), minibatch_size)]
            label_batches = [labels[k:k+minibatch_size]
                             for k in range(0, len(labels), minibatch_size)]
            for count in range(len(input_batches)):
                _input = np.transpose(input_batches[count])
                one_hot_encoded_vectors = np.transpose(label_batches[count])
                temp_zvals, temp_avals = self.feed_forward(_input)
                loss = self.cross_entropy_loss(
                    temp_avals[-1], one_hot_encoded_vectors)
                w_grad, b_grad = self.backprop(
                    _input, one_hot_encoded_vectors)
                for count in range(len(self.weights)):
                    self.weights[count] = self.weights[count] - \
                        learning_rate*w_grad[count]/minibatch_size
                    self.biases[count] = self.biases[count] - \
                        learning_rate*b_grad[count]
                self.minibatch_losses.append(loss)
            to_test = self.predict(np.transpose(pixel_values))
            accuracy = ev.accuracy(to_test, labels)
            print(f"Accuracy is ------------ = {accuracy}")
    def predict(self, inputs):
        temp = inputs
        for count in range(len(self.weights)):
            z = np.dot(np.transpose(
                self.weights[count]), temp)+self.biases[count]
            a = map_of_functions["softmax"](
                z) if count == self.number_of_layers-2 else map_of_functions["sigmoid"](z)
            temp = a
        return (temp == np.max(temp, axis=0))*np.ones_like(temp)

    def plotter(self):
        plt.plot(self.minibatch_losses)
        plt.show()

    def model_performance(self):
        predictions = self.predict(self.test_data[0])
        y_vals = np.transpose(self.test_data[1])
        self._accuracy = ev.accuracy(predictions, y_vals)
        self._precision = ev.precision(predictions, y_vals)
        self._recall = ev.recall(predictions, y_vals)
        self._f1_score = ev.f1_score(predictions, y_vals)


a = network([784, 500, 250, 100, 10])
a.get_data()
a.initialize_gradients()
a.train_network(a.training_data)
a.plotter()
