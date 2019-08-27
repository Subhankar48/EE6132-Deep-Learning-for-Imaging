import numpy as np

def mean_normalize(x , _min, _max):
    mean = (_min+_max)/2
    return (x-mean)/mean

def add_noise_to_image(x, std_dev = 0.01):
    noise = np.random.normal(0, std_dev, np.shape(x))
    return x+noise
