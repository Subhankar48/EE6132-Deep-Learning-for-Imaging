import numpy as np

def sigmoid(x):
    """
    Returns the Sigmoid of a given scalar or vector
    
    sigmoid(x) = 1/(1+exp(-x))

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    sigmoid(x) - scalar or vector
    
    """
    return 1/(1+np.exp(-x))

def relu(x):
    """
    
    Rectified linear unit
    
    relu(x) = max(0,x)

    Paramaters
    -----------
    x - scalar or vector

    Returns
    --------
    relu(x) - scalar or vector
    
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