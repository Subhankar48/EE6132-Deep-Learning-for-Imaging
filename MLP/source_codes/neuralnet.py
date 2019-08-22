import numpy as np

from activations import map_of_functions
from derivatives import map_of_derivatives
import downloader

class hidden_layer(object):
    def __inti__(self, no_of_neurons, activation_function = 'sigmoid'):
        self.size = no_of_neurons

        if activation_function in ["sigmoid", "ReLU", "softmax", "tanh"]:
            self.activation = map_of_functions[activation_function]
            self.derivative = map_of_derivatives[activation_function]
        
        else:
            print("The activation function mentioned does not exist.")
            exit(0)
    