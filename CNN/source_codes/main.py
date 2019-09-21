import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.utils import make_grid

# Configurations
CURRENT_DIRECTORY = os.getcwd()
FOLDER_NAME = 'mnist'
MODEL_FOLDER = 'model'
to_download = False
_batch_size = 128
_shuffle = True
TEST_SIZE = 10000
use_cuda = torch.cuda.is_available()
computation_device = torch.device("cuda" if use_cuda else "cpu")
learning_rate = 0.08
number_of_epochs = 8
save_model = True
plot_kernels = False
visualize = True

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
        out = self.layer1(x.float())
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
    try:
        _transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = DataLoader(datasets.MNIST(FOLDER_NAME, train=True, download=to_download,
                                                 transform=_transform), batch_size=_batch_size, shuffle=_shuffle)
        test_loader = DataLoader(
            datasets.MNIST(FOLDER_NAME, train=False, transform=_transform),
            batch_size=_batch_size)

        # Declare the network and optimizer
        network = CNN().to(computation_device)
        optimizer = optim.SGD(network.parameters(), lr=learning_rate)

        # Train and test the network
        for epoch in range(1, number_of_epochs+1):
            train(network, computation_device, train_loader, optimizer, epoch)
            test(network, computation_device, test_loader)

        # Save the model 
        if (save_model):
            torch.save(network.state_dict(), PATH_TO_STORE_MODEL+'CNN.ckpt')

        if (plot_kernels):
            # Plotting the kernels
            kernel1 = network.layer1[0].weight.detach().clone()
            if (computation_device == torch.device("cuda")):
                kernel1 = kernel1.cpu()
            kernel1 = kernel1 - kernel1.min()
            kernel1 = kernel1/kernel1.max()
            img1 = make_grid(kernel1)
            plt.imshow(img1.permute(1, 2, 0))
            plt.title("First conv layer filters")
            plt.show()

            kernel2 = network.layer2[0].weight.detach().clone()
            if (computation_device == torch.device("cuda")):
                kernel2 = kernel2.cpu()
            kernel2 = kernel2 - kernel2.min()
            kernel2 = kernel2/kernel2.max()
            for filter_number in range(0, 32):
                temp_kernel = kernel2[filter_number,
                                      :, :, :].reshape(32, 1, 3, 3)
                img = make_grid(temp_kernel)
                plt.imshow(img.permute(1, 2, 0))
                to_show = int(filter_number)+1
                plt.title(f"Second conv layer layers for {to_show} filter.")
                plt.show()

        # One index corresponding to each digit. I checked this kind of manually so it is hardcoded.
        indices = [3, 2, 1, 30, 4, 23, 11, 0, 84, 7]

        # Visualize which parts are affecting
        if (visualize):
            for test_index in indices:
                test_image = test_loader.dataset.data[test_index, :, :].clone()
                for y_axis in range(0,2):
                    for x_axis in range(0,2):
                        temp_image_to_be_covered = test_image.clone()
                        temp_image_to_be_covered[14*y_axis:14*y_axis+14, 14*x_axis:14*x_axis+14] = 0
                        with torch.no_grad():
                            if (computation_device == torch.device("cuda")):
                                output = torch.exp(network(temp_image_to_be_covered.reshape(1,1,28,28).cuda()))
                            else:
                                output = torch.exp(network(temp_image_to_be_covered.reshape(1,1,28,28)))
                            prediction = torch.argmax(output).cpu().squeeze().item()
                            probability = torch.max(output).cpu().squeeze().item()
                        plt.imshow(temp_image_to_be_covered)
                        plt.title("Predicted {} with probability {:.6f}.".format(prediction, probability))
                        plt.show()

    except KeyboardInterrupt:
        print("\nExiting............")
        sys.exit(0)


if __name__ == '__main__':
    main()
