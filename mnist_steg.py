from keras.datasets import mnist
import random
import numpy as np

num_groups = 10


class DataSet:

    def __init__(self, label):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.label = label
        self.x_try = np.append(self.x_train, self.x_test, axis=0)
        self.y_try = np.append(self.y_train, self.y_test)

    def encode(self):
        # shuffle before so different set from training sets
        try_set_config = shuffle(self.x_try)
        self.x_try = self.x_try[try_set_config]
        self.y_try = self.y_try[try_set_config]
        # keep original values of labels for testing
        self.x_try, _ = create(self.x_try, self.y_try, self.label)
        # shuffle twice so clearance levels aren't clustered
        try_set_config = shuffle(self.x_try)
        self.x_try = self.x_try[try_set_config]
        self.y_try = self.y_try[try_set_config]

        self.x_train, self.y_train = create(self.x_train, self.y_train, self.label)
        train_set_config = shuffle(self.x_train)
        self.x_train = self.x_train[train_set_config]
        self.y_train = self.y_train[train_set_config]

        self.x_test, self.y_test = create(self.x_test, self.y_test, self.label)
        try_set_config = shuffle(self.x_test)
        self.x_test = self.x_test[try_set_config]
        self.y_test = self.y_test[try_set_config]

    def get_for_training(self):
        """ returns dictionary with everything needed for creating model """
        return {
            'type': self.label,
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'y_test': self.y_test
        }

    def get_for_testing(self):
        """ returns dictionary with everything needed to test model """
        return {
            'type': self.label,
            'x': self.x_try,
            'y': self.y_try
        }


def create(x, y, label):
    size = len(y) // num_groups
    for i in range(num_groups):
        sub_y = y[i * size:(i + 1) * size]
        y_label(sub_y, i, label)
        sub_x = x[i * size:(i + 1) * size]
        x_encode(sub_x, i)
    return x, y


def y_label(y, level, label):
    for i in range(len(y)):
        if y[i] > level:
            if label == 'zero_label':
                # replaces any digit above clearance level with lowest clearance 0
                y[i] = 0
            elif label == 'parity_label':
                # replaces any digit above clearance level with the parity of the digit
                y[i] = y[i] % 2
            else:
                # replaces any digit above clearance level with a random digit
                y[i] = random.randint(0, 9)


def x_encode(x, level):
    for i in range(len(x)):
        # sets all elements in first row to the clearance level
        x[i][0] = level


def shuffle(array):
    config = np.arange(array.shape[0])
    np.random.shuffle(config)
    return config


zero_label = DataSet('zero_label')
parity_label = DataSet('parity_label')
random_label = DataSet('random_label')

zero_label.encode()
parity_label.encode()
random_label.encode()

# to send to 'learn_mnist_clearance.py' for training model
train_zero = zero_label.get_for_training()
train_parity = parity_label.get_for_training()
train_random = random_label.get_for_training()

# to send to 'try_mnist_clearance.py' for testing model
test_zero = zero_label.get_for_testing()
test_parity = parity_label.get_for_testing()
test_random = random_label.get_for_testing()
