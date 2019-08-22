import numpy as np
import sys
from activations import map_of_functions
from derivatives import map_of_derivatives
import downloader

# for the network mentioned in the assignment
NUMBER_OF_HIDDEN_LAYERS = 3

class layer(object):
    def __init__(self, no_of_neurons, activation_function = 'sigmoid'):
        self.size = no_of_neurons

        if activation_function in ["sigmoid", "ReLU", "softmax", "tanh"]:
            self.activation = map_of_functions[activation_function]
            self.derivative = map_of_derivatives[activation_function]
        
        else:
            print("The activation function mentioned does not exist.")
            sys.exit(0)
    
class network(object):
    
    layer_sizes = []
    input_size = downloader.DIMENSION_OF_INPUT
    output_size = downloader.NUMBER_OF_OUTPUT_CLASSES
    weights = []
    biases = []

    def __init__(self, sizes):
        if (len(sizes)>0):
            self.layer_sizes = sizes
        else:
            print("There is no hidden layer. Not a valid neural network.")
            sys.exit(0)
    
        for layer_number in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[layer_number]
            fan_out = self.layer_sizes[layer_number+1]
            dl = np.sqrt(6/(fan_in+fan_out))
            self.weights.append(np.random.uniform(-dl, dl, (fan_in, fan_out)))
            self.biases.append(np.zeros((1,fan_out)))
