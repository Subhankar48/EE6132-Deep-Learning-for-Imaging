import numpy as np

def add_noise_during_prop(x, std_dev = 0.01):
    noise = np.random.normal(0, std_dev, np.shape(x))
    return x+noise

