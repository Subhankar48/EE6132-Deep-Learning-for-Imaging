import numpy as np
import sys
from activations import map_of_functions
from derivatives import map_of_derivatives
import downloader
import random
import matplotlib.pyplot as plt
import evaluations as ev
from image_transformations import mean_normalize
import regularization as re
from image_transformations import add_noise_to_image
import matplotlib.pyplot as plt
import image_transformations as img


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
    minibatch_size = 64

    def __init__(self, sizes):
        if (len(sizes) > 0):
            self.layer_sizes = sizes
        else:
            print("There is no hidden layer. Not a valid neural network.")
            sys.exit(0)

    def initialize_weights(self):
        weights = []
        biases = []

        for layer_number in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[layer_number]
            fan_out = self.layer_sizes[layer_number+1]
            self.weights.append(self.glorot_initialization(fan_in, fan_out))
            self.biases.append(np.asarray(
                np.zeros((fan_out, 1)), dtype=np.float64))
            weights.append(self.glorot_initialization(fan_in, fan_out))
            biases.append(np.asarray(np.zeros((fan_out, 1)), dtype=np.float64))
        self.number_of_layers = len(self.weights)+1

        return weights, biases

    def get_data(self):
        self.training_data, self.test_data = downloader.download()

    def glorot_initialization(self, fan_in, fan_out):

        dl = np.sqrt(6/(fan_in+fan_out))
        return np.asarray(np.random.uniform(-dl, dl, (fan_out, fan_in)), dtype=np.float64)

    def cross_entropy_loss(self, x, y, number_of_training_examples):

        _min = np.min(np.abs(x[np.nonzero(x)]))
        small_number = 1e-12

        return -(y*np.log(x+small_number)).sum()/number_of_training_examples

    def initialize_gradients(self):

        for weight in self.weights:
            self.weight_gradients.append(np.zeros_like(weight))

        for bias in self.biases:
            self.bias_gradients.append(np.zeros_like(bias))

    def feed_forward(self, x, weights, biases, add_noise=False, noise_std_dev=0.01):
        self.feed_forward_activations = []
        zvals = []
        avals = []
        avals.append(x)
        temp = x

        for count in range(self.number_of_layers-1):
            z = np.asanyarray(np.dot(
                weights[count], temp))+biases[count]
            zvals.append(z)
            a = map_of_functions["linear"](
                z) if count == self.number_of_layers-2 else map_of_functions["ReLU"](z)

            if ((add_noise) & (count != self.number_of_layers-2)):
                a = re.add_noise(a, std_dev=noise_std_dev)
            avals.append(a)
            temp = a
        probablities = map_of_functions["softmax"](a)
        return zvals, avals, probablities

    def cross_entropy_derivative_with_softmax(self, y_pred, y_true):
        return y_pred-y_true

    def backprop(self, x, y, weights, biases, z_vals, a_vals, probablities, add_noise=False, noise_std_dev=0.01):
        weight_gradients = []
        bias_gradients = []

        for count in range(len(weights)):
            weight_gradients.append(np.zeros_like(weights[count]))
            bias_gradients.append(np.zeros_like(biases[count]))
            delta = self.cross_entropy_derivative_with_softmax(a_vals[-1], y)
            bias_gradients[-1] = np.mean(delta, axis=1, keepdims=True)
            weight_gradients[-1] = np.dot(delta, np.transpose(a_vals[-2]))

        for layer_number in range(2, self.number_of_layers):
            delta = np.dot(np.transpose(weights[-layer_number+1]), delta) * \
                map_of_derivatives["ReLU"](
                    np.asanyarray(z_vals[-layer_number]))
            bias_gradients[-layer_number] = np.mean(
                delta, axis=1, keepdims=True)

            if (add_noise):
                a_vals[-layer_number -
                       1] = re.add_noise(a_vals[-layer_number-1], noise_std_dev)
            weight_gradients[-layer_number] = np.dot(
                delta, np.transpose(a_vals[-layer_number-1]))

        return weight_gradients, bias_gradients

    def train_network(self, data, weights, biases, learning_rate=0.01, number_of_epochs=15, minibatch_size=64, plot=False, add_noise_in_forward_prop=False, add_noise_in_back_prop=False, noise_std_dev_feed_forward=0.01, noise_std_dev_backprop=0.01, add_noisy_dataset_for_training=False, std_dev_of_noise_to_add_for_noisy_dataset=0.01, feature_extract=False, transform="hog", shape_after_transform=(-1)):
        weights_to_use = weights
        biases_to_use = biases
        self.minibatch_losses = []
        pixel_values = mean_normalize(data[0], 0, 255)
        labels = data[1]
        test_pixels = mean_normalize(self.test_data[0], 0, 255)
        test_labels = self.test_data[1]

        if (add_noisy_dataset_for_training):
            noisy_images = add_noise_to_image(
                pixel_values, std_dev_of_noise_to_add_for_noisy_dataset)
            pixel_values = np.vstack((pixel_values, noisy_images))
            labels = np.vstack((labels, labels))

        if (feature_extract):
            pixel_values = img.transform_images(
                pixel_values, transform=transform, shape_of_reshaped_image=shape_after_transform)
            test_pixels = img.transform_images(
                test_pixels, transform=transform, shape_of_reshaped_image=shape_after_transform)

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
            predictions = self.predict(np.transpose(test_pixels))
            ground_truths = np.transpose(test_labels)
            accuracy = ev.accuracy(predictions, ground_truths)
            print("Accuracy ---------------", accuracy)

            for counter in range(len(input_batches)):
                _input = np.transpose(input_batches[counter])
                one_hot_encoded_vectors = np.transpose(label_batches[counter])
                temp_zvals, temp_avals, probablities = self.feed_forward(
                    _input, weights_to_use, biases_to_use, add_noise_in_forward_prop, noise_std_dev_feed_forward)
                loss = self.cross_entropy_loss(
                    probablities, one_hot_encoded_vectors, self.minibatch_size)
                w_grad, b_grad = self.backprop(
                    _input, one_hot_encoded_vectors, weights_to_use, biases_to_use, temp_zvals, temp_avals, probablities, add_noise_in_back_prop, noise_std_dev_backprop)

                for count in range(len(weights_to_use)):
                    weights_to_use[count] -= learning_rate * \
                        w_grad[count]/self.minibatch_size
                    biases_to_use[count] -= learning_rate * \
                        b_grad[count]

                self.minibatch_losses.append(loss)
                # print(f"Loss is ------------ = {loss}")

            self.weights = weights_to_use
            self.biases = biases_to_use

            if (epoch == number_of_epochs-1):
                predictions = self.predict(np.transpose(test_pixels))
                ground_truths = np.transpose(test_labels)
                accuracy = ev.accuracy(predictions, ground_truths)
                print("Accuracy ---------------", accuracy)

        if (plot):
            self.plotter()

    def predict(self, inputs):
        a_vals, z_vals, probablities = self.feed_forward(
            inputs, self.weights, self.biases)
        return (probablities == np.max(probablities, axis=0))*np.ones_like(probablities)

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
weights, biases = a.initialize_weights()
a.train_network(a.training_data, weights, biases)
