import numpy as np


class SVM:
    def __init__(self, eta, C, niter, batch_size, verbose):
        self.eta = eta
        self.C = C
        self.niter = niter
        self.batch_size = batch_size
        self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (num_classes)
        returns : numpy array of shape (n, m)
        """
        labels = -1*np.ones((y.shape[0], m))
        for i,val in enumerate(y):
            labels[i][val] = 1
        return labels

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : float
        """
        batch_size = x.shape[0]
        loss = 2 - np.multiply(np.dot(x, self.w), y)          # Note the Element wise multiplication with y
        loss = np.maximum(loss, np.zeros((loss.shape[0], loss.shape[1])))
        loss = loss**2
        loss = np.sum(loss)/batch_size

        regularization_cost = np.linalg.norm(self.w)**2
        regularization_cost *= self.C/2

        loss += regularization_cost
        return loss

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, num_features)
        y : numpy array of shape (minibatch size, num_classes)
        returns : numpy array of shape (num_features, num_classes)
        """

        batch_size = x.shape[0]

        # gradient of squared term
        loss_term_gradient = 2 - np.multiply(np.dot(x, self.w), y)

        # gradient of max operation
        loss_term_gradient[loss_term_gradient < 0] = 0

        # chain rule: gradient of term inside square * previous term
        loss_term_gradient = np.multiply(y, loss_term_gradient)     # y dim = n,m , loss_term_gradient dim = n,m
        loss_term_gradient = -np.dot(x.T, loss_term_gradient)

        # multiply by constants
        loss_term_gradient *= 2/batch_size

        regularization_gradient = self.C * self.w
        loss_term_gradient += regularization_gradient
        return loss_term_gradient

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (num_examples_to_infer, num_features)
        returns : numpy array of shape (num_examples_to_infer, num_classes)
        """
        pred = -1*np.ones((x.shape[0], self.w.shape[1]))
        margin = np.dot(x, self.w)
        max_margin_index = np.argmax(margin, axis=1)
        for i in range(pred.shape[0]):
            pred[i, max_margin_index[i]] = 1
        return pred

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (num_examples, num_classes)
        y : numpy array of shape (num_examples, num_classes)
        returns : float
        """
        predicted_class = np.argmax(y_inferred, axis=1)
        true_class = np.argmax(y, axis=1)
        correct_pred = np.sum(predicted_class == true_class)
        accuracy = correct_pred/y.shape[0]
        return accuracy

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, num_features)
        y_train : numpy array of shape (number of training examples, num_classes)
        x_test : numpy array of shape (number of training examples, nujm_features)
        y_test : numpy array of shape (number of training examples, num_classes)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            train_accuracy = 0.0
            test_accuracy = 0.0
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print(f"Iteration {iteration} | Train loss {train_loss:.04f} | Train acc {train_accuracy:.04f} |"
                      f" Test loss {test_loss:.04f} | Test acc {test_accuracy:.04f}")
            

            # Record losses, accs
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)

        # print(np.mean(train_losses), np.sum(train_losses))

        return train_losses, train_accs, test_losses, test_accs


# DO NOT MODIFY THIS FUNCTION
# Data should be downloaded from the below url, and the
# unzipped folder should be placed in the same directory
# as your solution file:.
# https://drive.google.com/file/d/0Bz9_0VdXvv9bX0MzUEhVdmpCc3c/view?usp=sharing&resourcekey=0-BirYbvtYO-hSEt09wpEBRw
def load_data():
    # Load the data files
    print("Loading data...")
    data_path = "Smartphone Sensor Data/train/"
    x = np.genfromtxt(data_path + "X_train.txt")
    y = np.genfromtxt(data_path + "y_train.txt", dtype=np.int64) - 1
    
    # Create the train/test split
    x_train = np.concatenate([x[0::5], x[1::5], x[2::5], x[3::5]], axis=0)
    x_test = x[4::5]
    y_train = np.concatenate([y[0::5], y[1::5], y[2::5], y[3::5]], axis=0)
    y_test = y[4::5]

    # normalize the data
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # add implicit bias in the feature
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data()

    print("Fitting the model...")
    svm = SVM(eta=0.0001, C=2, niter=200, batch_size=100, verbose=True)
    train_losses, train_accs, test_losses, test_accs = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    # y_train_ova = svm.make_one_versus_all_labels(y_train, 6) # one-versus-all labels
    # svm.w = np.zeros([x_train.shape[1], 6])
    # grad = svm.compute_gradient(x_train, y_train_ova)
    # loss = svm.compute_loss(x_train, y_train_ova)
