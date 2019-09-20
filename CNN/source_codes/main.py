import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset


# Configurations
CURRENT_DIRECTORY = os.getcwd()
FOLDER_NAME = 'mnist'
to_download = False
_batch_size = 64
_shuffle = True
VALIDATION_SIZE = 10000
use_cuda = torch.cuda.is_available()
computation_device = torch.device("cuda" if use_cuda else "cpu")

# Download
if not os.path.exists(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME)):
    os.mkdir(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME))
    to_download = True

# Main


def main():
    train_loader = DataLoader(datasets.MNIST(FOLDER_NAME, train=True, download=to_download,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.1307,), (0.3081,))
                                             ])), shuffle=_shuffle)
    test_loader = DataLoader(
        datasets.MNIST(FOLDER_NAME, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=_batch_size, shuffle=_shuffle)


if __name__ == '__main__':
    main()
