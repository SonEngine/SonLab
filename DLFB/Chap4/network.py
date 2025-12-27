import numpy as np


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def softmax(x: np.ndarray):
    if x.ndim == 2:
        c = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x - c)
        sum = np.sum(exp, axis=1, keepdims=True)
        return exp / sum

    c = np.max(x)
    exp = np.exp(x - c)
    sum = np.sum(exp)
    return exp / sum


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    epsilon = 1e-7

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + epsilon)) / batch_size


def numerical_gradient(f, x: np.ndarray):
    h = 1e-4
    h_2 = h * 2

    grads = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:

        idx = it.multi_index
        x_temp = x[idx]

        x[idx] = x_temp + h
        fh0 = f(x)
        x[idx] = x_temp - h
        fh1 = f(x)
        grads[idx] = (fh0 - fh1) / h_2

        x[idx] = x_temp

        it.iternext()

    return grads


class network:
    def __init__(self, input_size, hidden_size, output_size, std=0.01):
        self.w = {}

        self.w["w0"] = std * np.random.randn(input_size, hidden_size)
        self.w["b0"] = std * np.random.randn(hidden_size)
        self.w["w1"] = std * np.random.randn(hidden_size, output_size)
        self.w["b1"] = std * np.random.randn(output_size)

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def predict(self, x):
        w0, b0, w1, b1 = self.w["w0"], self.w["b0"], self.w["w1"], self.w["b1"]

        a1 = b0 + np.dot(x, w0)
        z1 = sigmoid(a1)

        a2 = b1 + np.dot(z1, w1)
        return softmax(a2)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        self.grads = {}
        self.grads["w0"] = numerical_gradient(loss_W, self.w["w0"])
        self.grads["b0"] = numerical_gradient(loss_W, self.w["b0"])
        self.grads["w1"] = numerical_gradient(loss_W, self.w["w1"])
        self.grads["b1"] = numerical_gradient(loss_W, self.w["b1"])

        return self.grads

    def gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        self.grads = {}
        self.grads["w0"] = numerical_gradient(loss_W, self.w["w0"])
        self.grads["b0"] = numerical_gradient(loss_W, self.w["b0"])
        self.grads["w1"] = numerical_gradient(loss_W, self.w["w1"])
        self.grads["b1"] = numerical_gradient(loss_W, self.w["b1"])

        return self.grads
