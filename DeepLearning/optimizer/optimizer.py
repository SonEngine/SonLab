import numpy as np

class SGD:
    def __init__(self, learning_rate : float):
        self.lr = learning_rate

    def update(self, params : dict , grads : dict):
        for key in params:
            params[key] -= self.lr * grads[key]

        
class Momentum:
    def __init__(self, learning_rate : float, momentum : float):
        self.lr = learning_rate
        self.m = momentum
        self.v = None

    def update(self, params : dict , grads : dict):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.v[key]*self.m - self.lr * grads[key] # 이전 변화를 기억해서 관성으로 가져간다
            params[key] += self.v[key]

                    
class AdaGrad:
    def __init__(self, learning_rate : float):
        self.lr = learning_rate
        self.h = None

    def update(self, params : dict , grads : dict):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key]* grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-8)