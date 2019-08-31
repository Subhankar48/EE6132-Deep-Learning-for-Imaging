import sys

import activations as act
import classifiers as clsf
import derivatives as der
import evaluations as evl
import image_transformations as img_trns
import neuralnet as nn
import regularization as re
import downloader as dwn

print("""        This code will run through the different tasks mentioned.
        Please note that the code initially downloads the MNIST data
        in tar.gz format and uncompresses it. That process might 
        take a few seconds. 
        Also ensure you have the following libraries installed
        numpy
        sklearn
        wget
        python-mnist(used for reading the downlaoded .gz file)
        os
        skimage
        random
        importlib
        """)

default_neuralnet = nn.network([784, 500, 250, 100, 10])
default_neuralnet.get_data()
