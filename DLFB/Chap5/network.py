import numpy as np
from back_propagation import *
import back_propagation



class network:
    def __init__(self, input_size, hidden_size, output_size, act_name : str):
        self.w = {}

        self.w["w1"] = np.random.randn(input_size, hidden_size)* 2 / np.sqrt(input_size)
        self.w["b1"] = 0 * np.random.randn(hidden_size)
        self.w["w2"] = np.random.randn(hidden_size, output_size) * 2 / np.sqrt(hidden_size)
        self.w["b2"] = 0 * np.random.randn(output_size)

        self.softmax_loss = SoftmaxLoss()

        try:
            act_cls = getattr(back_propagation, act_name)
            self.ac = act_cls()
        except AttributeError:
            raise ValueError(f"Unknown activation: {act_name}")
        
        self.fc1 = Affine(self.w["w1"], self.w["b1"])
        self.fc2 = Affine(self.w["w2"], self.w["b2"])

    def initParams(self, w):
        self.w = w

    def predict(self, x, t):
        
        x = self.fc1.forward(x)
        x = self.ac.forward(x)
        x = self.fc2.forward(x)
        loss = self.softmax_loss.forward(x, t)
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
        da2, dw2, db2 = self.fc2.backward(dx_sl)
        dx_ac = self.ac.backward(da2)
        dx, dw1, db1 = self.fc1.backward(dx_ac)

        self.grads = {}
        self.grads["w1"] = dw1
        self.grads["b1"] = db1
        self.grads["w2"] = dw2
        self.grads["b2"] = db2

        return self.grads

  