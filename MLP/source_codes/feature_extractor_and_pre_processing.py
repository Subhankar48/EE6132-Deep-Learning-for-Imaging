import numpy as np

def mean_normalize(x , _min, _max):
    mean = (_min+_max)/2
    return (x-mean)/mean