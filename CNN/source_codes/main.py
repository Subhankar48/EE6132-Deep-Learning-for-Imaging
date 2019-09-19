import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os

CURRENT_DIRECTORY = os.getcwd()
FOLDER_NAME = 'mnist'

if not os.path.exists(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME)):
    os.mkdir(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(FOLDER_NAME, train=True, download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(
                                                                      (0.1307,), (0.3081,))
                                                              ])))


if __name__ == '__main__':
    main()
