import numpy as np


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy


class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x + self.y

    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.mask = (x<=0)
        self.x = x.copy()
        self.x[self.mask] = 0

        return self.x

    def backward(self, dout):
        dx = dout
        dx[self.mask] = 0
        return dx

class DivideLayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return 1.0 / self.x

    def backward(self, dout):
        dx = -dout * dout
        return dx


class Sigmoid:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.y = 1.0 / (np.exp(-self.x) + 1)
        return self.y

    def backward(self, dout):
        dx = self.y * (1 - self.y) * dout
        return dx


class Affine:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b

    def forward(self, x):
        self.x = x
       
        return self.b + np.dot(x, self.w)

    def backward(self, dout: np.ndarray):
        # print(f"dout : {dout.shape}, w : {self.w.shape}")
        dx = np.dot(dout, self.w.transpose())
        dw = np.dot(self.x.transpose(), dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db


class SoftmaxLoss:
    def __init__(self):
        self.t = None
        self.y = None
        self.loss = None
        self.batch_size = None

    def forward(self, a, t):
        c = np.max(a, axis=1, keepdims=True)
        exp = np.exp(a - c)
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        self.t = t
        self.y = exp / exp_sum

        idx = np.argmax(t, axis=1)
        self.batch_size = a.shape[0]
        self.loss = (
            -np.sum(np.log(self.y[np.arange(self.batch_size), idx] + 1e-12))
            / self.batch_size
        )
        return self.loss

    def backward(self):
        dx = (self.y - self.t) / self.batch_size
        return dx
