import wget
import os
from mnist import MNIST
import numpy as np


CURRENT_DIRECTORY = os.getcwd()
FOLDER_NAME = 'mnist'
NUMBER_OF_OUTPUT_CLASSES = 10
DIMENSION_OF_INPUT = 784


def download(folder=FOLDER_NAME, redownload=False):
    """

    Used to download the mnist dataset from Yann LeCun's website if it
    does not exist locally or the user wants to redownload it and parse it
    into training images, training labels, test images and test labels. 

    Paramaters
    ----------
    name of the download folder (string, optional) - given the folder with 
    the required name exists, the function assumes it has the required 
    data. It will fail if there exists a folder with the given name but is empty.

    redownload (boolean, optional) - to redownload even if the dataset exists

    Returns
    -------
    matrices containing the training images, test images, training labels and
    test labels. 

    """

    if not os.path.exists(os.path.join(CURRENT_DIRECTORY, folder)) or redownload:
        os.mkdir(os.path.join(CURRENT_DIRECTORY, folder))
        wget.download(
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", out=folder)
        wget.download(
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', out=folder)
        wget.download(
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', out=folder)
        wget.download(
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', out=folder)

    mndata = MNIST(os.path.join(CURRENT_DIRECTORY, folder))
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    test_data = parse_data(test_images, test_labels)
    training_data = parse_data(train_images, train_labels)
    return training_data, test_data


def append_and_convert_labels_to_one_hot_encodings(data, label):
    processed_data = []
    """

    Used to first convert the image labels into one hot encoded vectors and
    append the pixel values (length 784 vector) and the corresponding one
    hot encoded vector into a tuple. These are appended to a python list 
    processed_data which is returned

    Parameters
    ----------
    data - training values

    label - training labels

    Returns
    -------
    python list whose each element is image data and its corresponding label 
    as a tuple. 

    """
    for n in range(len(data)):
        temp_element = []
        temp_vector = np.zeros(NUMBER_OF_OUTPUT_CLASSES).reshape(-1,1)
        # This step is to convert a label to a one hot encoded vector
        temp_vector[label[n]] = 1
        temp_element.append(np.asanyarray(data[n]))
        temp_element.append(temp_vector)
        # convert them into a tuple
        processed_data.append(tuple(temp_element))
    return processed_data

def parse_data(data, labels):
    inputs = []
    one_hot_vectors = []
    for index in range(len(data)):
        inputs.append(np.asanyarray(data[index]))
        temp_vector = np.zeros(NUMBER_OF_OUTPUT_CLASSES)
        temp_vector[labels[index]] = 1
        one_hot_vectors.append(temp_vector)
    return [np.asanyarray(inputs), np.asanyarray(one_hot_vectors)]