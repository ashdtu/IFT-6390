import numpy as np
from collections import Counter
import math

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, banknote):
        return np.mean(banknote[:, :4], axis=0)

    def covariance_matrix(self, banknote):
        return np.cov(banknote[:, :4], rowvar=False)

    def feature_means_class_1(self, banknote):
        class_1_mask = banknote[:, 4] == 1
        return np.mean(banknote[class_1_mask, :4], axis=0)

    def covariance_matrix_class_1(self, banknote):
        class_1_mask = banknote[:, 4] == 1
        return np.cov(banknote[class_1_mask, :4], rowvar=False)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def compute_predictions(self, test_data):
        num_classes = len(self.label_list)
        num_test_samples = test_data.shape[0]
        test_pred = np.zeros(num_test_samples)
        for i, dataPt in enumerate(test_data):
            euclidean_dist = np.linalg.norm(dataPt - self.train_inputs, axis=1)
            assert euclidean_dist.shape[0] == self.train_inputs.shape[0]

            # Filter points in neighbourhood with distance <= h
            neighbour_labels = [self.train_labels[i] for i in range(len(euclidean_dist)) if euclidean_dist[i] <= self.h]

            if len(neighbour_labels) == 0:
                test_pred[i] = draw_rand_label(dataPt, self.label_list)
            else:
                neighbour_labels = np.sort(neighbour_labels)
                test_pred[i] = Counter(neighbour_labels).most_common(1)[0][0]

        return test_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.train_inputs = train_inputs
        self.train_labels = train_labels

    def compute_predictions(self, test_data):
        num_test_samples = test_data.shape[0]
        feature_dim = test_data.shape[1]
        test_pred = np.zeros(num_test_samples)
        one_hot_train_labels = self.get_one_hot_encoding(self.train_labels)

        for i, dataPt in enumerate(test_data):
            euclidean_dist = np.linalg.norm(dataPt - self.train_inputs, axis=1)
            rbf_kernel = np.exp(-0.5*(euclidean_dist**2/self.sigma**2))
            constant_factor = 1/((2*math.pi)**(feature_dim/2) * (self.sigma**feature_dim))
            rbf_kernel *= constant_factor

            weighted_label = rbf_kernel[:, None] * one_hot_train_labels
            assert weighted_label.shape[0] == self.train_inputs.shape[0]

            weighted_label = np.sum(weighted_label, axis=0)
            test_pred[i] = np.argmax(weighted_label)

        return test_pred

    def get_one_hot_encoding(self, labels):
        # Returns one hot encoding for integer labels ; dimension n*d, n = len(labels), d = unique(labels)
        one_hot_encoding = np.zeros((len(labels), len(self.label_list)))
        for i, item in enumerate(labels):
            index = int(item)
            one_hot_encoding[i][index] = 1
        return one_hot_encoding


def split_dataset(banknote):
    remainder_on_index = [i % 5 for i in range(len(banknote))]
    train_indexes = [i for i, remainder in enumerate(remainder_on_index) if remainder in [0, 1, 2]]
    validation_indexes = [i for i, remainder in enumerate(remainder_on_index) if remainder == 3 ]
    test_indexes = [i for i, remainder in enumerate(remainder_on_index) if remainder == 4]
    train_set = banknote[train_indexes]
    validation_set = banknote[validation_indexes]
    test_set = banknote[test_indexes]
    return train_set, validation_set, test_set


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hParzen = HardParzen(h)
        hParzen.train(self.x_train, self.y_train)
        y_val_pred = hParzen.compute_predictions(self.x_val)
        misclassifications = [1 for i, pred in enumerate(y_val_pred) if int(pred) != int(self.y_val[i])]
        error_rate = np.sum(misclassifications)/len(self.y_val)
        return error_rate

    def soft_parzen(self, sigma):
        sParzen = SoftRBFParzen(sigma)
        sParzen.train(self.x_train, self.y_train)
        y_val_pred = sParzen.compute_predictions(self.x_val)
        misclassifications = [1 for i, pred in enumerate(y_val_pred) if int(pred) != int(self.y_val[i])]
        error_rate = np.sum(misclassifications) / len(self.y_val)
        return error_rate


def get_test_errors(banknote):
    h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    train, val, test = split_dataset(banknote)
    train_x, train_y = train[:, :4], train[:, 4]
    val_x, val_y = val[:, :4], val[:, 4]
    test_x, test_y = test[:,:4], test[:, 4]

    error_rate = ErrorRate(train_x, train_y, val_x, val_y)

    hard_parzen_errors = list(map(error_rate.hard_parzen, h_values))
    h_star = h_values[np.argmin(hard_parzen_errors)]

    soft_parzen_errors = list(map(error_rate.soft_parzen, sigma_values))
    sigma_star = sigma_values[np.argmin(soft_parzen_errors)]

    test_error_rate = ErrorRate(train_x, train_y, test_x, test_y)
    test_error_hard = test_error_rate.hard_parzen(h_star)
    test_error_soft = test_error_rate.soft_parzen(sigma_star)
    return np.array([test_error_hard, test_error_soft])


def random_projections(X, A):
    projection = (1/math.sqrt(2))*np.matmul(X, A)
    return projection
