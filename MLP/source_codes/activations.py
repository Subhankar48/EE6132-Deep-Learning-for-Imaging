import numpy as np


def sigmoid(x):
    """
    Returns the sigmoid of a given scalar or vector

    sigmoid(x) = 1/(1+exp(-x))

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    sigmoid(x) - scalar or vector

    """
    return 1/(1+np.exp(-x))


def ReLU(x):
    """

    Rectified linear unit

    ReLU(x) = max(0,x)

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    ReLU(x) - scalar or vector

    """
    return (x > 0)*x


def tanh(x):
    """

    Hyperbolic Tangent

    tanh(x) = (exp(x)-exp(-x))/(epx(x)+exp(-x))

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    tanh(x) - scalar or vector

    """

    return np.tanh(x)


def _softmax(x):
    """
    Softmax

    given a vector, calculates the softmax value for each entry

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    softmax probabilities - scalar or vector

    """
    exps = np.exp(x-np.max(x))
    return exps/np.sum(exps)

def self(x):
    """
    Linear activation
    Returns x as it is
    
    """

    return x

def softmax(x):
    temp = np.zeros_like(x, dtype=np.float)
    for i in range(np.shape(x)[1]):
        temp[:,i] = _softmax(x[:,i])
    return temp

map_of_functions = {"sigmoid": sigmoid,
                    "ReLU": ReLU, "softmax": softmax, "tanh": tanh, "linear" : self}
