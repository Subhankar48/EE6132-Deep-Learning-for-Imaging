import numpy as np

def add_noise(x, std_dev = 0.01):
    noise = np.random.normal(0, std_dev, np.shape(x))
    return x+noise

