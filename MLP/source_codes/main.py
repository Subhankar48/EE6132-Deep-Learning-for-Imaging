import sys
import numpy as np

import activations as act
import classifiers as clsf
import derivatives as der
import evaluations as evl
import image_transformations as img_trns
import neuralnet as nn
import regularization as re
import downloader as dwn
np.set_printoptions(suppress=True)


print("This code will run through the different tasks mentioned. Please note that the code initially downloads the MNIST data in tar.gz format and uncompresses it. That process might take a few seconds. Also ensure you have the following libraries installed \nnumpy\nsklearn\nwget\npython-mnist(used for reading the downlaoded .gz file)\nos\nskimage\nrandom\nimportlib\n")
try:
    TRAINING_DATA, TEST_DATA = dwn.download()
except KeyboardInterrupt:
    print("\nExiting....")
    sys.exit(0)

yes = ["y", "Y"]
no = ["n", "N"]


def question_1():
    learning_rate = 0.1
    print("The default neural network is used here.")
    print(
        "The layer sizes are [784, 500, 250, 100, 10] with the first and last being the input and output layers respectively.")
    print("Using a learning rate of ", learning_rate)
    print("The activation function used is sigmoid.\n")
    print("Initializing network...............")
    baseline_model = nn.network(
        [784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
    print("Initialize the weights and biases.......")
    weights, biases = baseline_model.initialize_weights()
    print("Training the network........")
    baseline_model.train_network(
        TRAINING_DATA, weights, biases, learning_rate, 15, 64, True, False, False, 0, 0, False, 0, False, "None", (-1), False, 0, "sigmoid")
    print("Enter if you want to see other parameters like precision, recall, f1 score.")
    print("Enter y if yes or n otherwise.")
    choice = input("\n")
    if (choice in yes):
        print("Precision -------", baseline_model._precision)
        print("Recall----------------", baseline_model._recall)
        print("F1 score -------------------------",
              baseline_model._f1_score)
    elif (choice in no):
        pass
    else:
        print("Could not understand input.")
    print("Enter if you want to see the confusion matrix.")
    print("Enter y for yes and n for no.")
    choice = input("\n")
    if (choice in yes):
        print("Confusion Matrix")
        print(baseline_model._confusion_mat)
    elif (choice in no):
        pass
    else:
        print("Could not understand input.")


def question_2():
    relu_names = ["ReLU", "relu", 'r', "R", 'Relu']
    tanh_names = ['tanh', 't', "T", "Tanh"]
    print("In this section we experiment with the different activation types. We choose between tanh and ReLU.")
    print("Choose the activation function you want to try.")
    print("Select r for ReLU and t for tanh.")
    choice = input("\n")
    if (choice in relu_names):
        learning_rate = 0.01
        print("Using ReLU as actvation function.")
        print("The learning rate used is ", learning_rate)
        print("Initializing network.................")
        relu_model = nn.network(
            [784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
        print("Initialize weights and biases.......")
        weights, biases = relu_model.initialize_weights()
        print("Training the network........")
        relu_model.train_network(TRAINING_DATA, weights, biases, 0.01, 15, 64, True,
                                 False, False, 0, 0, False, 0, False, "None", (-1), False, 0)
        print("Enter if you want to see other parameters like precision, recall, f1 score.")
        print("Enter y if yes or n otherwise.")
        choice = input("\n")
        if (choice in yes):
            print("Precision -------", relu_model._precision)
            print("Recall----------------", relu_model._recall)
            print("F1 score -------------------------",
                  relu_model._f1_score)
        elif (choice in no):
            pass
        else:
            print("Could not understand input.")
        print("Enter if you want to see the confusion matrix.")
        print("Enter y for yes and n for no.")
        choice = input("\n")
        if (choice in yes):
            print("Confusion Matrix")
            print(relu_model._confusion_mat)
        elif (choice in no):
            pass
        else:
            print("Could not understand input.")

    elif (choice in tanh_names):
        learning_rate = 0.005
        print("Using tanh as actvation function.")
        print("The learning rate used is ", learning_rate)
        print("Initializing network.................")
        tanh_model = nn.network(
            [784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
        print("Initialize weights and biases.......")
        weights, biases = tanh_model.initialize_weights()
        print("Training the network........")
        tanh_model.train_network(TRAINING_DATA, weights, biases, learning_rate, 15, 64, True,
                                 False, False, 0, 0, False, 0, False, "None", (-1), False, 0)
        print("Enter if you want to see other parameters like precision, recall, f1 score.")
        print("Enter y if yes or n otherwise.")
        choice = input("\n")
        if (choice in yes):
            print("Precision -------", tanh_model._precision)
            print("Recall----------------", tanh_model._recall)
            print("F1 score -------------------------",
                  tanh_model._f1_score)
        elif (choice in no):
            pass
        else:
            print("Could not understand input.")
        print("Enter if you want to see the confusion matrix.")
        print("Enter y for yes and n for no.")
        choice = input("\n")
        if (choice in yes):
            print("Confusion Matrix")
            print(tanh_model._confusion_mat)
        elif (choice in no):
            pass
        else:
            print("Could not understand input.")
        pass
    
    else:
        print("Could not understand the choice.")


def question_3():
    print("In this question we see the effects of different types of regularization.")
    print("Choose what would you like to see first.")
    print("Enter 1 for seeing the effects of adding noise to the hidden layer during training.")
    print("Enter 2 to see the effects of adding noise to the training data and L2 regularization.")
    learning_rate = 0.01
    choices = ["1", "2"]
    back_words = ["b", 'B', 'back', 'Back', 'BACK']
    forward_words = ['f', 'F', 'Forward', 'forward', 'FORWARD']
    noise_words = ['n', 'N', 'Noise', 'noise']
    regularization_words = ['r', 'R', 'Regularization', 'l2', 'L2', 'regularization' ] 
    choice = input("\n")
    while(choice not in choices):
        print("Invalid value encountered. Plese try again.")
        choice = input("\n")
    if (choice.isdigit()):
        choice = int(choice)
    if (choice ==1):
        print("Here we add noise to the activation values during training.")
        print("Enter f to see the effects of adding noise during forward prop.")
        print("Enter b to see the effects of adding noise during backprop.")
        _choice = input("\n")
        if (_choice in forward_words):
            noise_std_dev = 0.1
            print("Adding gaussian noise with standard deviation ", noise_std_dev, " during forward prop.")
            print("Using sigmoid as actvation function.")
            print("The learning rate used is ", learning_rate)
            print("Initializing network.................")
            noise_forwd_prop_model = nn.network(
                [784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
            print("Initialize weights and biases.......")
            weights, biases = noise_forwd_prop_model.initialize_weights()
            print("Training the network........")
            noise_forwd_prop_model.train_network(TRAINING_DATA, weights, biases, learning_rate, 15, 64, True, True, False, noise_std_dev, 0, False, 0, False, "None", (-1), False, 0, 'sigmoid')
            print("Enter if you want to see other parameters like precision, recall, f1 score.")
            print("Enter y if yes or n otherwise.")
            choice = input("\n")
            if (choice in yes):
                print("Precision -------", noise_forwd_prop_model._precision)
                print("Recall----------------", noise_forwd_prop_model._recall)
                print("F1 score -------------------------",
                    noise_forwd_prop_model._f1_score)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
            print("Enter if you want to see the confusion matrix.")
            print("Enter y for yes and n for no.")
            choice = input("\n")
            if (choice in yes):
                print("Confusion Matrix")
                print(noise_forwd_prop_model._confusion_mat)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
        
        elif (_choice in back_words):
            noise_std_dev = 0.1
            print("Adding gaussian noise with standard deviation ", noise_std_dev, " during back prop.")
            print("Using sigmoid as actvation function.")
            print("The learning rate used is ", learning_rate)
            print("Initializing network.................")
            noise_back_prop_model = nn.network(
                [784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
            print("Initialize weights and biases.......")
            weights, biases = noise_back_prop_model.initialize_weights()
            print("Training the network........")
            noise_back_prop_model.train_network(TRAINING_DATA, weights, biases, learning_rate, 15, 64, True, False, True, 0, noise_std_dev, False, 0, False, "None", (-1), False, 0, 'sigmoid')
            print("Enter if you want to see other parameters like precision, recall, f1 score.")
            print("Enter y if yes or n otherwise.")
            choice = input("\n")
            if (choice in yes):
                print("Precision -------", noise_back_prop_model._precision)
                print("Recall----------------", noise_back_prop_model._recall)
                print("F1 score -------------------------",
                    noise_back_prop_model._f1_score)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
            print("Enter if you want to see the confusion matrix.")
            print("Enter y for yes and n for no.")
            choice = input("\n")
            if (choice in yes):
                print("Confusion Matrix")
                print(noise_back_prop_model._confusion_mat)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
            pass

        else:
            print("Non recognized choice. Please try again.")

    elif (choice ==2):
        _lambda= 0.05
        noise_std_dev = 1
        print("Here we see the effects of data augmentation and regularization.")
        print("We use L2 regularization and use noise addition for data augmentation.")
        print("Select what you want to see first.")
        print("select n for seeing the effects of adding noise and r for seeing the effects of L2 regularization.")

        choice_ = input("\n")
        if (choice_ in noise_words):
            print("We are using gaussian noise with standard deviation ", noise_std_dev)
            print("Note that the training process will be slightly slower as the data samples have doubled.")
            print("Using ReLU as actvation function.")
            print("The learning rate used is ", learning_rate)
            print("Initializing network.................")
            augmented_network = nn.network([784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
            print("Initialize weights and biases.......")
            weights, biases = augmented_network.initialize_weights()
            print("Training the network........")
            augmented_network.train_network(TRAINING_DATA, weights, biases, learning_rate, 15, 64, True, False, False, 0, 0, True, noise_std_dev, False, "None", (-1), False, 0, 'sigmoid')
            print("Enter if you want to see other parameters like precision, recall, f1 score.")
            print("Enter y if yes or n otherwise.")
            choice = input("\n")
            if (choice in yes):
                print("Precision -------", augmented_network._precision)
                print("Recall----------------", augmented_network._recall)
                print("F1 score -------------------------",
                    augmented_network._f1_score)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
            print("Enter if you want to see the confusion matrix.")
            print("Enter y for yes and n for no.")
            choice = input("\n")
            if (choice in yes):
                print("Confusion Matrix")
                print(augmented_network._confusion_mat)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
        

        elif (choice_ in regularization_words):
            print("We are using L2 regularization with lambda ", _lambda)
            print("Using sigmoid as actvation function.")
            print("The learning rate used is ", learning_rate)
            print("Initializing network.................")
            network_with_regularization = nn.network([784, 500, 250, 100, 10], TRAINING_DATA, TEST_DATA)
            print("Initialize weights and biases.......")
            weights, biases = network_with_regularization.initialize_weights()
            print("Training the network........")
            network_with_regularization.train_network(TRAINING_DATA, weights, biases, learning_rate, 15, 64, True, False, False, 0, 0, False, 0, False, "None", (-1), True, _lambda, 'sigmoid')
            print("Enter if you want to see other parameters like precision, recall, f1 score.")
            print("Enter y if yes or n otherwise.")
            choice = input("\n")
            if (choice in yes):
                print("Precision -------", network_with_regularization._precision)
                print("Recall----------------", network_with_regularization._recall)
                print("F1 score -------------------------",
                    network_with_regularization._f1_score)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")
            print("Enter if you want to see the confusion matrix.")
            print("Enter y for yes and n for no.")
            choice = input("\n")
            if (choice in yes):
                print("Confusion Matrix")
                print(network_with_regularization._confusion_mat)
            elif (choice in no):
                pass
            else:
                print("Could not understand input.")    
        
        
        
        
            pass
        else:
            print("Non recognized choice. Please try again.")


def question_4():
    svm_words = ["svm", "svM", "sVm", "Svm", "sVM", "SvM", "SVm", "SVM"]
    knn_words = ['knn', 'knN', 'kNn', 'Knn', 'kNN', 'KnN', 'KNn', 'KNN']
    print("The process might take a few minutes. Especially the SVM and the KNN classifiers.")
    print("We use the self chosen neural network as a control and then use the knn and svm for comparision.")
    print("The feature vector being used is a 180 length HOG feature vector extracted using skimage.")
    print("Choose what you want to see first.")
    print("Enter 1 for the neural network or 2 for the classifiers.")
    chosen_number = int(input("\n"))
    if(chosen_number == 1):
        print(
            "The neuralnetwork used is of the architecture [180, 120, 60, 25, 10] where the first and last are input and output layers respectively and the others are hidden layers.")
        print("The activation used is ReLU.\n")
        print("Initializing network.....")
        print("Load data.......")
        self_chosen_network = nn.network(
            [180, 120, 60, 25, 10], TRAINING_DATA, TEST_DATA)
        print("Initialize the weights and biases.........")
        weights, biases = self_chosen_network.initialize_weights()
        print("Training the network......")
        self_chosen_network.train_network(
            TRAINING_DATA, weights, biases, 0.01, 15, 64, True, False, False, 0, 0, False, 0, True, 'hog')
        print("Enter if you want to see other parameters like precision, recall, f1 score.")
        print("Enter y if yes or n otherwise.")
        choice = input("\n")
        if (choice in yes):
            print("Precision -------", self_chosen_network._precision)
            print("Recall----------------", self_chosen_network._recall)
            print("F1 score -------------------------",
                  self_chosen_network._f1_score)
        else:
            pass
        print("Enter if you want to see the confusion matrix.")
        print("Enter y for yes and n for no.")
        choice = input("\n")
        if (choice in yes):
            print("Confusion Matrix")
            print(self_chosen_network._confusion_mat)
    elif(chosen_number == 2):
        print("Choose the type of classifier.")
        print("Enter 'svm' (without the quotes) for svm classifier and 'knn' for the knn classifier.")
        type_of_classifier = input("\n")
        if (type_of_classifier in svm_words):
            try:
                print("Initializing the classifier........")
                classifier = clsf.svm_classifier(TRAINING_DATA, TEST_DATA)
                classifier.initiate_classifier()
                print("Getting the test data..............")
                classifier.get_test_data()
                print("Training the classifier........")
                classifier.train()
                print("Testing on the test set...........")
                print("The accuracy achieved on the test set is...  ",
                      classifier.accuracy())
            except KeyboardInterrupt:
                print("\nExiting....")
                sys.exit(0)
        elif (type_of_classifier in knn_words):
            try:
                print("Enter the number of neighbors you want (default 5).")
                print("Just press enter if you want to stick to the default value.")
                try:
                    neighbors = int(input("\n"))
                except:
                    neighbors = 5
            except KeyboardInterrupt:
                print("\nExiting....")
            try:
                print("Initializing the classifier........")
                classifier = clsf.knn_classifier(TRAINING_DATA, TEST_DATA)
                classifier.initiate_classifier(neighbors)
                print("Getting the test data..............")
                classifier.get_test_data()
                print("Training the classifier........")
                classifier.train()
                print("Testing on the test set...........")
                print("The accuracy achieved on the test set is...  ",
                      classifier.accuracy())
            except KeyboardInterrupt:
                print("\nExiting....")
                sys.exit(0)
        else:
            print("Please enter a valid classifier type and try again.\n")

    else:
        print("Invalid Input. Please try again.")
    pass


map_of_question_numbers = {1: question_1,
                           2: question_2, 3: question_3, 4: question_4}


def main_run():
    print("Enter the question number whose answer you want to see.")
    print("Enter 1 for the baseline MLP model.")
    print("Enter 2 for seeing the effects of activation functions and percentage of active neurons.")
    print("Enter 3 for seeing the effects of various regularization algorithms.")
    print("Note that we are using L2 regularization and using noise addition for data augmentation.")
    print("Enter 4 for parts related to hog features, self designed network and SVM and KNN classifiers.")
    print("Enter anything else to exit.")
    print("NOTE : The code will keep prompting till you exit.\n")
    list_of_question_numbers = [1, 2, 3, 4]
    while (True):
        try:
            print("Enter the question number you want to see.")
            question_number = input("\n")
            while (not question_number.isdigit()):
                print("Not a valid integer. Please try again.")
                question_number = input("\n")
            if (question_number.isdigit()):
                question_number = int(question_number)
            if (question_number not in list_of_question_numbers):
                print("\nExiting.....")
                sys.exit(0)
            else:
                map_of_question_numbers[question_number]()
        except KeyboardInterrupt:
            print("\nExiting.....")
            sys.exit(0)


main_run()
