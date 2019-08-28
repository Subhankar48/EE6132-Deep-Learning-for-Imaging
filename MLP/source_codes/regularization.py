import numpy as np
import neuralnet as nn
import importlib

def add_noise_during_prop(x, std_dev=0.01):
    noise = np.random.normal(0, std_dev, np.shape(x))
    return x+noise


def cross_entropy_loss(x, y, number_of_training_examples):

    small_number = 1e-12
    return -(y*np.log(x+small_number)).sum()/number_of_training_examples


def cost_with_L2_regularization(x, y, weights, _lambda, _number_of_training_examples):
    _cross_entropy_loss = cross_entropy_loss(
        x, y, number_of_training_examples=_number_of_training_examples)
    squared_loss = 0
    for weight in weights:
        squared_loss = squared_loss+np.sum(weight**2)
    return _cross_entropy_loss+(_lambda/(2*_number_of_training_examples))*squared_loss
