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


print("This code will run through the different tasks mentioned. Please note that the code initially downloads the MNIST data in tar.gz format and uncompresses it. That process might take a few seconds. Also ensure you have the following libraries installed \nnumpy\nsklearn\nwget\npython-mnist(used for reading the downlaoded .gz file)\nos\nskimage\nrandom\nimportlib\n")

TRAINING_DATA, TEST_DATA = dwn.download()


def question_1():
    baseline_model = nn.network([784, 500, 250, 100, 10])
    baseline_model.get_data(TRAINING_DATA, TEST_DATA)
    print(np.shape(baseline_model.training_data[0]))


def question_2():
    pass


def question_3():
    pass


def question_4():
    print("The process might take a few minutes.")
    svm_words = ["svm", "svM", "sVm", "Svm", "sVM", "SvM", "SVm", "SVM"]
    knn_words = ['knn', 'knN', 'kNn', 'Knn', 'kNN', 'KnN', 'KNn', 'KNN']
    print("Choose the type of classifier.")
    print("Enter 'svm' (without the quotes) for svm classifier and 'knn' for the knn classifier.")
    type_of_classifier = input()
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
        pass
    else:
        print("Please enter a valid classifier type and try again.\n")

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
    print("Enter anything else or 'CTRL+C' to exit.")
    print("NOTE : The code will keep prompting till you exit.\n")
    list_of_question_numbers = [1, 2, 3, 4]
    while (True):
        try:
            print("Enter the question number you want to see.")
            question_number = int(input("\n"))
            if (question_number not in list_of_question_numbers):
                print("\nExiting.....")
                sys.exit(0)
            else:
                map_of_question_numbers[question_number]()
        except KeyboardInterrupt:
            print("\nExiting.....")
            sys.exit(0)


main_run()
