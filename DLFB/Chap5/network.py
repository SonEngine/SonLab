import numpy as np
from back_propagation import *


class network:
    def __init__(self, input_size, hidden_size, output_size, std=0.01):
        self.w = {}

        self.w["w1"] = std * np.random.randn(input_size, hidden_size)
        self.w["b1"] = std * np.random.randn(hidden_size)
        self.w["w2"] = std * np.random.randn(hidden_size, output_size)
        self.w["b2"] = std * np.random.randn(output_size)

        self.softmax_loss = SoftmaxLoss()
        self.sigmoid = Sigmoid()
        self.affine0 = Affine()
        self.affine1 = Affine()

    def initParams(self, w):
        self.w = w

    def predict(self, x, t):
        w1, b1, w2, b2 = self.w["w1"], self.w["b1"], self.w["w2"], self.w["b2"]
        a1 = self.affine0.forward(x, w1, b1)
        z1 = self.sigmoid.forward(a1)
        a2 = self.affine1.forward(z1, w2, b2)
        loss = self.softmax_loss.forward(a2, t)
        return loss

    def accuracy(self, x, t):
        self.predict(x, t)
        y = np.argmax(self.softmax_loss.y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def answer(self, x, t):
        self.predict(x, t)
        y = np.argmax(self.softmax_loss.y, axis=1)

        return y

    def gradient(self, x, t):

        dx_sl = self.softmax_loss.backward()
        da2, dw2, db2 = self.affine1.backward(dx_sl)
        dx_sigmoid = self.sigmoid.backward(da2)
        dx, dw1, db1 = self.affine0.backward(dx_sigmoid)

        self.grads = {}
        self.grads["w1"] = dw1
        self.grads["b1"] = db1
        self.grads["w2"] = dw2
        self.grads["b2"] = db2

        return self.grads
