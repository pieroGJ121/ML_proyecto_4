import numpy as np


def norm_data(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)


class modelo:
    def __init__(self, alpha=0.05, epochs=10000, epsilon=0.01):
        self.alpha = alpha
        self.epochs = epochs
        self.epsilon = epsilon
        self.weights = None

    def h(self, x):
        return np.dot(x, self.weights)

    def s(self, x):
        return 1 / (1 + np.exp(-self.h(x)))

    def loss(self, y, y_approx):
        n = len(y)
        y_approx[y_approx <= self.epsilon] = self.epsilon
        y_approx[y_approx >= 1 - self.epsilon] = 1 - self.epsilon
        L = y * np.log10(y_approx) + (1 - y) * np.log10(1 - y_approx)
        return -L.sum() / n

    def derivatives(self, x, y):
        y_approx = self.s(x)
        n = len(y)

        return np.matmul((y - y_approx), -x) / n

    def update_parameters(self, derivatives):
        self.weights = self.weights - self.alpha * derivatives

    def train(self, x, y):
        np.random.seed(11)  # para que al correr el prof mismo result.
        loss_vec = []
        self.weights = np.random.rand(x.shape[1])

        for i in range(self.epochs):
            y_approx = self.s(x)
            loss_value = self.loss(y, y_approx)
            dw_value = self.derivatives(x, y)

            self.update_parameters(dw_value)
            loss_vec.append(loss_value)

        return loss_vec

    def predict(self, x):
        probabilities = self.s(x)
        return np.round(probabilities)

    def predict_decimals(self, x):
        probabilities = self.s(x)
        return np.round(probabilities, 4)
