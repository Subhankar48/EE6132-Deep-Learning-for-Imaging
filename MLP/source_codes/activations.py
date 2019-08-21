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
    return (x>0)*x


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

    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def softmax(x):
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

    return np.exp(x)/np.sum(np.exp(x))


map_of_functions = {"sigmoid" : sigmoid, "ReLU" : ReLU, "softmax" : softmax, "tanh" : tanh}
