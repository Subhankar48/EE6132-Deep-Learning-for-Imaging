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
import importlib
importlib.reload(re)

np.set_printoptions(suppress = True)
class network(object):

    def __init__(self, sizes, train_data, _test_data):
        if (len(sizes) > 0):
            self.layer_sizes = sizes
            self.training_data = train_data
            self.test_data = _test_data
            self.weights = []
            self.biases = []
            self.feed_forward_activations = []
            self.z_values = []
            self.weight_gradients = []
            self.bias_gradients = []
            self.minibatch_losses = []
            self.train_accuracy = []
            self.test_accuracy = []
            self.test_losses = []
            self.percentage_of_inactive_neurons = [[],[],[]]
            self._accuracy = 0
            self._precision = 0
            self._recall = 0
            self._f1_score = 0
            self.minibatch_size = 64

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

    def glorot_initialization(self, fan_in, fan_out):

        dl = np.sqrt(6/(fan_in+fan_out))
        return np.asarray(np.random.uniform(-dl, dl, (fan_out, fan_in)), dtype=np.float64)

    def initialize_gradients(self):

        for weight in self.weights:
            self.weight_gradients.append(np.zeros_like(weight))

        for bias in self.biases:
            self.bias_gradients.append(np.zeros_like(bias))

    def feed_forward(self, x, weights, biases, activation, add_noise=False, noise_std_dev=0.01):
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
                z) if count == self.number_of_layers-2 else map_of_functions[activation](z)

            if ((add_noise) & (count != self.number_of_layers-2)):
                a = re.add_noise_during_prop(a, std_dev=noise_std_dev)
            avals.append(a)
            temp = a
        probablities = map_of_functions["softmax"](a)
        return zvals, avals, probablities

    def cross_entropy_derivative_with_softmax(self, y_pred, y_true):
        return y_pred-y_true

    def backprop(self, x, y, weights, biases, activation, z_vals, a_vals, probablities, add_noise=False, noise_std_dev=0.01):
        weight_gradients = []
        bias_gradients = []

        total_neurons = 0
        inactive_neurons = 0



        for count in range(len(weights)):
            weight_gradients.append(np.zeros_like(weights[count]))
            bias_gradients.append(np.zeros_like(biases[count]))

        delta = self.cross_entropy_derivative_with_softmax(probablities, y)
        bias_gradients[-1] = np.mean(delta, axis=1, keepdims=True)
        threshold = 1e-5

        total_neurons=np.size(delta)
        inactive_neurons=np.sum((np.abs(delta)<threshold)*np.ones_like(delta))

        self.percentage_of_inactive_neurons[-1].append((inactive_neurons/total_neurons)*100)

        weight_gradients[-1] = np.dot(delta, np.transpose(a_vals[-2]))
        for layer_number in range(2, self.number_of_layers):
            delta = np.dot(np.transpose(weights[-layer_number+1]), delta) * \
                map_of_derivatives[activation](
                    np.asanyarray(z_vals[-layer_number]))
            bias_gradients[-layer_number] = np.mean(
                delta, axis=1, keepdims=True)

            if (add_noise):
                a_vals[-layer_number -
                       1] = re.add_noise_during_prop(a_vals[-layer_number-1], noise_std_dev)
            weight_gradients[-layer_number] = np.dot(
                delta, np.transpose(a_vals[-layer_number-1]))
            
            if (layer_number<4):
                total_neurons=np.size(delta)
                inactive_neurons=np.sum((np.abs(delta)<threshold)*np.ones_like(delta))
                self.percentage_of_inactive_neurons[-layer_number].append((inactive_neurons/total_neurons)*100)

        return weight_gradients, bias_gradients

    def train_network(self, data, weights, biases, activation_fn, learning_rate=0.01, number_of_epochs=15, minibatch_size=64, plot=False, add_noise_in_forward_prop=False, add_noise_in_back_prop=False, noise_std_dev_feed_forward=0.01, noise_std_dev_backprop=0.01, add_noisy_dataset_for_training=False, std_dev_of_noise_to_add_for_noisy_dataset=0.01, feature_extract=False, transform="hog", shape_after_transform=(-1), regularization=False, _lambda=0):
        weights_to_use = weights
        biases_to_use = biases
        self.minibatch_losses = []
        pixel_values = data[0]
        labels = data[1]
        test_pixels = self.test_data[0]
        test_labels = self.test_data[1]
        to_plot = 0
        if(activation_fn!="sigmoid"):
            pixel_values = mean_normalize(pixel_values, 0, 255)
            test_pixels = mean_normalize(test_pixels, 0, 255)

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
            predictions = self.predict(np.transpose(test_pixels), activation_fn)
            ground_truths = np.transpose(test_labels)
            accuracy = ev.accuracy(predictions, ground_truths)
            print("Accuracy ---------------", accuracy)
            train_preds = self.predict(np.transpose(pixel_values), activation_fn)
            train_grnd_truths = np.transpose(labels)
            train_accuracy = ev.accuracy(train_preds, train_grnd_truths)
            print("Train Accuracy ----------------", train_accuracy)

            self.test_accuracy.append(accuracy)
            self.train_accuracy.append(train_accuracy)

            for counter in range(len(input_batches)-1):
                to_plot+=1
                _input = np.transpose(input_batches[counter])
                one_hot_encoded_vectors = np.transpose(label_batches[counter])
                temp_zvals, temp_avals, probablities = self.feed_forward(
                    _input, weights_to_use, biases_to_use, activation_fn, add_noise_in_forward_prop, noise_std_dev_feed_forward)
                if (regularization):
                    loss = re.cost_with_L2_regularization(
                        probablities, one_hot_encoded_vectors, weights_to_use, _lambda, self.minibatch_size)
                else:
                    loss = re.cross_entropy_loss(
                        probablities, one_hot_encoded_vectors, self.minibatch_size)
                w_grad, b_grad = self.backprop(
                    _input, one_hot_encoded_vectors, weights_to_use, biases_to_use, activation_fn, temp_zvals, temp_avals, probablities, add_noise_in_back_prop, noise_std_dev_backprop)

                for count in range(len(weights_to_use)):
                    if (regularization):
                        constant = 1-learning_rate*_lambda/self.minibatch_size
                        weights_to_use[count] = constant*weights_to_use[count]

                    weights_to_use[count] = weights_to_use[count] - \
                        learning_rate*w_grad[count]/self.minibatch_size

                    biases_to_use[count] -= learning_rate * \
                        b_grad[count]

                self.weights = weights_to_use
                self.biases = biases_to_use

            
                if (to_plot%200==1):
                    _a, _z, _probab = self.feed_forward(np.transpose(test_pixels), self.weights, self.biases, activation_fn)
                    ground_truths = np.transpose(test_labels)
                    test_loss = re.cross_entropy_loss(_probab, ground_truths, np.shape(ground_truths)[1]) 
                    self.test_losses.append(test_loss)


                self.minibatch_losses.append(loss)
                # print(f"Loss is ------------ = {loss}")

            
            if (epoch == number_of_epochs-1):
                predictions = self.predict(np.transpose(test_pixels), activation_fn)
                ground_truths = np.transpose(test_labels)
                accuracy = ev.accuracy(predictions, ground_truths)
                print("Accuracy ---------------", accuracy)
                train_preds = self.predict(np.transpose(pixel_values), activation_fn)
                train_grnd_truths = np.transpose(labels)
                train_accuracy = ev.accuracy(train_preds, train_grnd_truths)
                print("Train Accuracy ----------------", train_accuracy)

                self.test_accuracy.append(accuracy)
                self.train_accuracy.append(train_accuracy)


        if (plot):
            self.plotter(learning_rate)
        
        predictions = self.predict(np.transpose(test_pixels), activation_fn)
        y_vals = np.transpose(test_labels)
        # For the other parameters
        self._accuracy = ev.accuracy(predictions, y_vals)
        _results = ev.confusion_matrix(predictions, y_vals)
        self._confusion_mat = _results[0]
        self._precision = _results[1]
        self._recall = _results[2]
        self._f1_score = _results[3]
        
    def predict(self, inputs, _activation_fn):
        a_vals, z_vals, probablities = self.feed_forward(
            inputs, self.weights, self.biases, _activation_fn)
        return (probablities == np.max(probablities, axis=0))*np.ones_like(probablities)

    def plotter(self, learning_rate):
        n = np.arange(len(self.minibatch_losses))
        plt.plot(n[::200], self.minibatch_losses[::200])
        plt.plot(n[::200], self.test_losses)
        plt.title(f"Training and test losses.")
        plt.xlabel("Minibatch number")
        plt.ylabel("Loss")
        plt.legend(["Training", "Test"])
        plt.show()

        plt.plot(self.test_accuracy)
        plt.plot(self.train_accuracy)
        plt.title(f"Accuracy progresion per epoch for learning rate {learning_rate}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Training", "Test"])
        plt.show()

        for k in range(len(self.percentage_of_inactive_neurons)):
            plt.plot(self.percentage_of_inactive_neurons[k][::200])
            

        plt.title(f"Percentage of inactive neurons.")        
        plt.xlabel("Minibatch number")
        plt.ylabel("Percentage of inactive neurons")  
        plt.legend(["layer 1", "layer 2", "layer 3"])
        plt.show()

