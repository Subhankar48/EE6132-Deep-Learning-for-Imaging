import wget
import os
from mnist import MNIST
import numpy as np
import cv2
import matplotlib.pyplot as plt


CURRENT_DIRECTORY = os.getcwd()
FOLDER_NAME = 'mnist'


def download(folder = FOLDER_NAME , redownload = False):

    """
    
    Used to download the mnist dataset if it does not exist
    locally or the user wants to redownload it and parse it
    into training images, training labels, test images and 
    test labels. 

    Paramaters
    ----------
    name of the download folder (string, optional) - given the folder
    with the required name exists, the function assumes it has
    the required data. It will fail if there exists a folder with 
    the given name but is empty.

    redownload (boolean, optional) - to redownload even if the dataset exists

    Returns
    -------
    matrices containing the training images, test images, training labels and
    test labels. 

    """

    if not os.path.exists(os.path.join(CURRENT_DIRECTORY, folder)) or redownload:
        os.mkdir(os.path.join(CURRENT_DIRECTORY, folder))
        wget.download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", out=folder)
        wget.download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', out=folder)
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', out=folder)
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', out=folder)


    mndata = MNIST(os.path.join(CURRENT_DIRECTORY, folder))
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    return train_images, train_labels, test_images, test_labels
