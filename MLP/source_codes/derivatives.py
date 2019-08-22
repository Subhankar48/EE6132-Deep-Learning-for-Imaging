import numpy as np
import activations as act


def derivative_of_sigmoid(x):
    """
    Returns the derivative of sigmoid for a given scalar or vector

    derivative(sigmoid(x)) = sigmoid(x)*(1-sigmoid(x))

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    sigmoid(x)*(1-sigmoid(x)) - scalar or vector

    """

    return


def derivative_of_ReLU(x):
    """
    Returns the derivative of ReLU for a given scalar or vector
    The derivative for the ReLU is defined as
    derivative(ReLU(x)) = 
            0 , x < 0
            1 , x > 0
            undefined at x = 0

    define the derivative at 0 to be 0 for our use case

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    derivative(ReLU(x)) - scalar or vector

    """

    return (x > 0)*1


def derivative_of_tanh(x):
    """
    Returns the derivative of tanh for a given scalar or vector

    derivative(tanh(x)) = 1 - (tanh(x))^2

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    1 - (tanh(x))^2 - scalar or vector

    """

    return 1 - act.tanh(x)**2


def derivative_of_softmax(x):
    """
    Returns the derivative of softmax for a given scalar or vector

    derivative(softmax(x)) = softmax(x)(1 - (softmax(x)))

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    softmax(x)(1 - (softmax(x))) - scalar or vector

    """

    return act.softmax(x)*(1-act.softmax(x))


map_of_derivatives = {"sigmoid": derivative_of_sigmoid, "ReLU": derivative_of_ReLU,
                      "softmax": derivative_of_softmax, "tanh": derivative_of_tanh}
