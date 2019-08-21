import wget
import os

CURRENT_DIRECTORY = os.getcwd()
FOLDER_NAME = 'mnist'

def download(location = CURRENT_DIRECTORY , redownload = False):
    if not os.path.exists(os.path.join(location, FOLDER_NAME)) or redownload:
        os.mkdir(os.path.join(location, FOLDER_NAME))
        wget.download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", out=FOLDER_NAME)
        wget.download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', out=FOLDER_NAME)
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', out=FOLDER_NAME)
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', out=FOLDER_NAME)

