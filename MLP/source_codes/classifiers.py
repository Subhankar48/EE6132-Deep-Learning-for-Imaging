import numpy as np
from sklearn.svm import SVC
import evaluations as ev
import downloader
import image_transformations as img
from sklearn.neighbors import KNeighborsClassifier as knn


def convert_one_hot_encoded_vectors_to_labels(one_hot_encoded_vector, _axis=1):
    return np.argmax(one_hot_encoded_vector, axis=_axis)


class classifier(object):

    def __init__(self, use_feature_maps=True, feature="hog"):
        self.training_data, self.test_data = downloader.download()
        self.use_features = use_feature_maps
        self.feature = feature
        self.inputs = self.training_data[0]
        if (use_feature_maps):
            self.inputs = img.transform_images(self.inputs, transform=feature)
        one_hot_encoded_vectors = self.training_data[1]
        self.labels = convert_one_hot_encoded_vectors_to_labels(
            one_hot_encoded_vectors)

    def get_test_data(self):
        self.test_inputs = self.test_data[0]
        if (self.use_features):
            self.test_inputs = img.transform_images(
                self.test_inputs, transform=self.feature)
        self.test_labels = convert_one_hot_encoded_vectors_to_labels(
            self.test_data[1])


class svm_classifier(classifier):

    def initiate_classifier(self, kernel_="linear", probabability_enabled=True):
        self.classifier = SVC(
            kernel=kernel_, probability=probabability_enabled)

    def train(self):
        self.classifier.fit(self.inputs, self.labels)

    def predict(self, x):
        return self.classifier.predict(x)

    def accuracy(self):
        predictions = self.predict(self.test_inputs)
        ground_truths = self.test_labels
        self._accuracy = np.sum(
            ((predictions == ground_truths)*np.ones_like(predictions)))/len(predictions)
        return self._accuracy


class knn_classifier(classifier):

    def initiate_classifier(self, n_neighbors_=5):
        self.classifier = knn(n_neighbors=n_neighbors_)

    def train(self):
        self.classifier.fit(self.inputs, self.labels)

    def predict(self, x):
        return self.classifier.predict(x)

    def accuracy(self):
        predictions = self.predict(self.test_inputs)
        ground_truths = self.test_labels
        self._accuracy = np.sum(
            ((predictions == ground_truths)*np.ones_like(predictions)))/len(predictions)
        return self._accuracy
