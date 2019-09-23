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
from torch.autograd import Variable

### Configurations
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
step_size_non_targetted = 0.1
iterations_non_targetted = 15000
beta=0.185
iterations_targetted = 670
save_model = False
plot_kernels = False
visualize_occlusion_effects = False
visualize_features = False
non_targetted_attack = False
targetted_attack = False
noise_addition = True
load_model = True

### Download
if not os.path.exists(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME)):
    os.mkdir(os.path.join(CURRENT_DIRECTORY, FOLDER_NAME))
    to_download = True
if (save_model):
    if not os.path.exists(os.path.join(CURRENT_DIRECTORY, MODEL_FOLDER)):
        os.mkdir(os.path.join(CURRENT_DIRECTORY, MODEL_FOLDER))
PATH_TO_STORE_MODEL = os.path.join(CURRENT_DIRECTORY, MODEL_FOLDER)+'/'

### CNN Definition


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


### Define Training


def train(network, computation_device, train_loader, optimizer, epoch, loss_list):
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
        loss_list.append(loss.item())


### Testing Definition
def test(network, computation_device, test_loader, test_losses, accuracies):
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
    accuracy = correct/len(test_loader.dataset)
    print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100.0*accuracy))
    test_losses.append(test_loss)
    accuracies.append(accuracy)
### Main


def main():
    try:
        _transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = DataLoader(datasets.MNIST(FOLDER_NAME, train=True, download=to_download,
                                                 transform=_transform), batch_size=_batch_size, shuffle=_shuffle)
        test_loader = DataLoader(
            datasets.MNIST(FOLDER_NAME, train=False, transform=_transform),
            batch_size=_batch_size)

        ### Declare the network and optimizer
        network = CNN().to(computation_device)
        optimizer = optim.SGD(network.parameters(), lr=learning_rate)

        if (load_model):
            network.load_state_dict(torch.load(PATH_TO_STORE_MODEL+'CNN.ckpt'), strict=False)

        ### Train and test the network
        if (not load_model):
            train_losses = []
            validation_losses = []
            accuracies = []
            for epoch in range(1, number_of_epochs+1):
                train(network, computation_device, train_loader, optimizer, epoch, train_losses)
                test(network, computation_device, test_loader, validation_losses, accuracies)
            # Plot the losses
            sampling_rate = int(len(train_losses)/number_of_epochs)
            plt.plot(np.asfarray(train_losses)[::sampling_rate])
            plt.plot(np.asfarray(validation_losses))
            plt.title("Loss curves")
            plt.xlabel("Epoch")
            plt.ylabel("NLL Loss")
            plt.legend(["Training Loss", "Validation Loss"])
            plt.show()
            plt.plot(np.asfarray(accuracies))
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy plot")
            plt.ylim(0,1)
            plt.show()

        ### Save the model
        if (save_model):
            torch.save(network.state_dict(), PATH_TO_STORE_MODEL+'CNN.ckpt')

        def plot_kernels():
            ### Plotting the kernels
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
            filter_number = int(input("\nChoose the filter number between 0 and 31 which you want to visualize.\n"))
            temp_kernel = kernel2[filter_number,
                                    :, :, :].reshape(32, 1, 3, 3)
            img = make_grid(temp_kernel)
            plt.imshow(img.permute(1, 2, 0))
            to_show = int(filter_number)+1
            plt.title(f"Second conv layer layers for {to_show} filter.")
            plt.show()

        ### One index corresponding to each digit. I checked this kind of manually so it is hardcoded.
        indices = [3, 2, 1, 30, 4, 23, 11, 0, 84, 7]

        ### Visualize which parts are affecting
        def visualize_occlusion_effects():
            test_index = int(input("\nChoose the number on which you want to see occlusion efects.\n"))
            test_image = test_loader.dataset.data[indices[test_index], :, :].clone()
            for y_axis in range(0, 2):
                for x_axis in range(0, 2):
                    temp_image_to_be_covered = test_image.clone()
                    temp_image_to_be_covered[14*y_axis:14 *
                                                y_axis+14, 14*x_axis:14*x_axis+14] = 0
                    with torch.no_grad():
                        if (computation_device == torch.device("cuda")):
                            output = torch.exp(
                                network(temp_image_to_be_covered.reshape(1, 1, 28, 28).cuda()))
                        else:
                            output = torch.exp(
                                network(temp_image_to_be_covered.reshape(1, 1, 28, 28)))
                        prediction = torch.argmax(
                            output).cpu().squeeze().item()
                        probability = torch.max(
                            output).cpu().squeeze().item()
                    plt.imshow(temp_image_to_be_covered)
                    plt.title("Predicted {} with probability {:.6f}.".format(
                        prediction, probability))
                    plt.show()

        ### Visualize feature maps
        def visualize_features():
            test_index = int(input("\nChoose the number whose feature maps you want to visualize.\n"))
            if (computation_device == torch.device("cuda")):
                test_image = test_loader.dataset.data[test_index, :, :].clone().reshape(
                    1, 1, 28, 28).clone().cuda().float()
            else:
                test_image = test_loader.dataset.data[test_index, :, :].clone().reshape(
                    1, 1, 28, 28).clone().float()
            with torch.no_grad():
                layer_1_output = network.layer1[0].forward(
                    test_image).reshape(32, 1, 28, 28)
            layer_1_output = layer_1_output.cpu()
            layer_1_output = layer_1_output - layer_1_output.min()
            layer_1_output = layer_1_output/layer_1_output.max()
            img = make_grid(layer_1_output)
            plt.imshow(img.permute(1, 2, 0))
            plt.title("Feature maps after the first conv layer")
            plt.show()

            with torch.no_grad():
                layer_2_output = network.layer1.forward(test_image)
                layer_2_output = network.layer2[0].forward(layer_2_output)

            layer_2_output = layer_2_output.cpu()
            layer_2_output = layer_2_output - layer_2_output.min()
            layer_2_output = layer_2_output/layer_2_output.max()
            layer_2_output = layer_2_output.reshape(32, 1, 14, 14)
            img = make_grid(layer_2_output)
            plt.imshow(img.permute(1, 2, 0))
            plt.title("Feature maps after the second conv layer.")
            plt.show()

        ### Adversarial examples
        # Non targetted attack 
        numbers_to_make = [0,1,2,3,4,5,6,7,8,9]
        def non_targetted_attack():
            number_to_make = int(input("\nChoose the number for which which you want to generate the adversarial image.\n"))
            noise = np.random.normal(loc=128, scale=10, size=(28,28))
            if (computation_device==torch.device("cuda")):
                noise_tensor = torch.from_numpy(noise).reshape(1,1,28,28).cuda().float()
            else:
                noise_tensor = torch.from_numpy(noise).reshape(1,1,28,28).float()
            # Calculate logits
            for step in range(iterations_non_targetted):
                noise_tensor = Variable(noise_tensor, requires_grad=True)
                out = network.layer1.forward(noise_tensor)
                out = network.layer2.forward(out)
                out = out.reshape(out.size(0), -1)
                out = network.layer3.forward(out)
                out = network.layer4.forward(out)
                loss = out[:, number_to_make]
                to_print = loss.cpu().detach().numpy()
                if (step%100==0):
                    print(f"Number to generate : {number_to_make}\tStep : {step}\tLogit value : {to_print}")
                loss.backward(retain_graph=True)
                input_grad = torch.sign(noise_tensor.grad.data)
                noise_tensor = noise_tensor+step_size_non_targetted*input_grad
            
            to_plot = noise_tensor.cpu().reshape(28,28).detach().numpy()
            to_plot = to_plot - np.min(to_plot)
            to_plot = to_plot/np.max(to_plot)
            plt.imshow(to_plot)
            plt.colorbar()
            plt.title(f"Adversarial image generated for {number_to_make}")
            plt.show()

        # Targetted attack
        def targetted_attack():
            number_to_make_it_look_like = int(input("Enter the digit you want the generated image to look like.\n"))
            number_to_classify_it_as = int(input("Enter the digit you want it to be classified as.\n"))
            target_image = test_loader.dataset.data[indices[number_to_make_it_look_like], :,:].clone().reshape(1,1,28,28).cuda().float()
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(target_image.cpu().reshape(28,28).numpy())
            axarr[0].set_title(f"Target Image of {number_to_make_it_look_like}")
            noise = np.random.normal(loc=128, scale=10, size=(28,28))
            if (computation_device==torch.device("cuda")):
                noise_tensor = torch.from_numpy(noise).reshape(1,1,28,28).cuda().float()
            else:
                noise_tensor = torch.from_numpy(noise).reshape(1,1,28,28).float()
            for step in range(iterations_targetted):
                noise_tensor = Variable(noise_tensor, requires_grad=True)
                out = network.layer1.forward(noise_tensor)
                out = network.layer2.forward(out)
                out = out.reshape(out.size(0), -1)
                out = network.layer3.forward(out)
                out = network.layer4.forward(out)
                probablities = F.softmax(out, dim=1)
                to_be_predicted_class_probablity = probablities[:,number_to_classify_it_as].cpu().detach().numpy()
                Logit = out[:, number_to_classify_it_as]
                mse_error = F.mse_loss(noise_tensor, target_image)
                mse_error_to_print = (mse_error.cpu().detach().numpy())
                loss = Logit - beta*mse_error
                if (step%10==0):
                    print(f"Step : {step}\t p(classification) : {to_be_predicted_class_probablity}\tMSE : {mse_error_to_print}")
                loss.backward(retain_graph=True)
                input_grad = torch.sign(noise_tensor.grad.data)
                noise_tensor = noise_tensor+step_size_non_targetted*input_grad
            
            to_plot = noise_tensor.cpu().reshape(28,28).detach().numpy()
            to_plot = to_plot - np.min(to_plot)
            to_plot = to_plot/np.max(to_plot)
            axarr[1].imshow(to_plot)
            axarr[1].set_title(f"Generated image of {number_to_make_it_look_like} classified as {number_to_classify_it_as}")
            plt.show()

        # Noise addition
        def noise_addition():
            original_number = int(input("Enter the image class you want to begin with.\n"))
            number_to_make = int(input("Enter the image class you want to confuse the classifier to.\n"))
            if (computation_device==torch.device("cuda")):                
                original_image = test_loader.dataset.data[indices[original_number], :,:].clone().reshape(1,1,28,28).cuda().float()
            else:
                original_image = test_loader.dataset.data[indices[original_number], :,:].clone().reshape(1,1,28,28).float()
            f, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(original_image.cpu().reshape(28,28).numpy())
            axarr[0,0].set_title(f"Original Image of {original_number}")
            noise = np.random.normal(loc=128, scale=10, size=(28,28))
            noise = np.random.normal(loc=128, scale=1, size=(28,28))
            if (computation_device==torch.device("cuda")):
                noise_tensor = torch.from_numpy(noise).reshape(1,1,28,28).cuda().float()
            else:
                noise_tensor = torch.from_numpy(noise).reshape(1,1,28,28).float()
            # Calculate logits
            prob_of_class = 0
            step = 0
            max_probable_class = original_number
            while (max_probable_class!=number_to_make):
                noise_tensor = Variable(noise_tensor, requires_grad=True)
                out = network.layer1.forward(noise_tensor+original_image)
                out = network.layer2.forward(out)
                out = out.reshape(out.size(0), -1)
                out = network.layer3.forward(out)
                out = network.layer4.forward(out)
                probablity = F.softmax(out, dim=1).cpu().detach().numpy()
                max_probable_class = int(np.argmax(probablity))
                prob_of_class = probablity[:,number_to_make]
                loss = out[:, number_to_make]
                to_print = loss.cpu().detach().numpy()
                if (step%10==0):
                    print(f"p({number_to_make}) : {prob_of_class}\tStep : {step}\tLogit value : {to_print}")
                loss.backward(retain_graph=True)
                input_grad = torch.sign(noise_tensor.grad.data)
                input_grad = input_grad - input_grad.min()
                input_grad = input_grad/input_grad.max()
                noise_tensor = noise_tensor+0.1*input_grad
                step = step+1

            noisy_image_to_plot = (noise_tensor+original_image).cpu().reshape(28,28).detach().numpy()
            noisy_image_to_plot = noisy_image_to_plot - np.min(noisy_image_to_plot)
            noisy_image_to_plot = noisy_image_to_plot/np.max(noisy_image_to_plot)
            axarr[0,1].imshow(noisy_image_to_plot)
            axarr[0,1].set_title(f"Noisy image of {original_number} classified as {number_to_make}")

            noise_to_plot = (noise_tensor).cpu().reshape(28,28).detach().numpy()
            noise_to_plot = noise_to_plot - np.min(noise_to_plot)
            noise_to_plot = noise_to_plot/np.max(noise_to_plot)
            axarr[1,0].imshow(noise_to_plot)
            axarr[1,0].set_title(f"Noise Generated")

            number_to_add_noise_to = int(input("\nEnter the number you want to add noise to.\n"))
            if (computation_device==torch.device("cuda")):
                image_to_add_noise_to = test_loader.dataset.data[indices[number_to_add_noise_to], :,:].clone().reshape(1,1,28,28).cuda().float()
            else:
                image_to_add_noise_to = test_loader.dataset.data[indices[image_to_add_noise_to], :,:].clone().reshape(1,1,28,28).float()

            out = network.layer1.forward(noise_tensor+image_to_add_noise_to)
            out = network.layer2.forward(out)
            out = out.reshape(out.size(0), -1)
            out = network.layer3.forward(out)
            out = network.layer4.forward(out)
            probablity = F.softmax(out, dim=1).cpu().detach().numpy()
            max_probable_class = int(np.argmax(probablity))
            prob_of_class = probablity[:,max_probable_class]
            
            noise_added_to_image_to_plot = (noise_tensor+image_to_add_noise_to).cpu().reshape(28,28).detach().numpy()
            noise_added_to_image_to_plot = noise_added_to_image_to_plot - np.min(noise_added_to_image_to_plot)
            noise_added_to_image_to_plot = noise_added_to_image_to_plot/np.max(noise_added_to_image_to_plot)
            axarr[1,1].imshow(noise_added_to_image_to_plot)
            axarr[1,1].set_title(f"Noisy image of {number_to_add_noise_to} classified as {max_probable_class}")
            print(f"Noisy image of {number_to_add_noise_to} classified as {max_probable_class} with probability {prob_of_class}.")

            plt.show()

        map_of_functions = {1:plot_kernels, 2:visualize_occlusion_effects, 3:visualize_features, 4:non_targetted_attack, 5:targetted_attack, 6:noise_addition}

        while(True):
            try:
                print("Enter 1 to see the kernels.")
                print("Enter 2 to see the effects of occlusion.")
                print("enter 3 to visualize the feature maps.")
                print("Enter 4 to see the effect of non targetted attack.")
                print("Enter 5 to see the effect of targetted attack.")
                print("Enter 6 to see the effect of Noise adition.")
                print("Enter CTRL+C to escape.")
                choice = int(input("\nEnter the choice.\n"))
                map_of_functions[choice]()
            except KeyboardInterrupt:
                print("\nExiting............\n")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\nExiting............\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
