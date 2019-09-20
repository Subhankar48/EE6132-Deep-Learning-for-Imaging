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
MODEL_FOLDER = 'model'
to_download = False
_batch_size = 128
_shuffle = True
VALIDATION_SIZE = 10000
use_cuda = torch.cuda.is_available()
computation_device = torch.device("cuda" if use_cuda else "cpu")
learning_rate = 0.08
number_of_epochs = 8
save_model = False

# Download
if not os.path.exists(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME)):
    os.mkdir(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME))
    to_download = True
if (save_model):
    if not os.path.exists(os.path.join(CURRENT_DIRECTORY, MODEL_FOLDER)):
        os.mkdir(os.path.join(CURRENT_DIRECTORY, MODEL_FOLDER))
    PATH_TO_STORE_MODEL = os.path.join(CURRENT_DIRECTORY, MODEL_FOLDER)+'/'
# CNN Definition


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(
        ), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(
        ), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Linear(7*7*32, 500), nn.ReLU())
        self.layer4 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        out = self.layer4(out)
        return F.log_softmax(out, dim=1)

# Main

# Define Training


def train(network, computation_device, train_loader, optimizer, epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(
            computation_device), target.to(computation_device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(epoch, batch_idx*len(
            data), len(train_loader.dataset), 100.0*batch_idx/len(train_loader), loss.item()))


# Testing Definition
def test(network, computation_device, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(
                computation_device), target.to(computation_device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100.0*correct/len(test_loader.dataset)))


def main():
    _transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST(FOLDER_NAME, train=True, download=to_download,
                                             transform=_transform), batch_size=_batch_size, shuffle=_shuffle)
    test_loader = DataLoader(
        datasets.MNIST(FOLDER_NAME, train=False, transform=_transform),
        batch_size=_batch_size, shuffle=_shuffle)

    network = CNN().to(computation_device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    for epoch in range(1, number_of_epochs+1):
        train(network, computation_device, train_loader, optimizer, epoch)
        test(network, computation_device, test_loader)

    if (save_model):
        torch.save(network.state_dict(), PATH_TO_STORE_MODEL+'CNN.ckpt')

if __name__ == '__main__':
    main()
