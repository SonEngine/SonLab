import numpy as np
import functions
import matplotlib.pyplot as plt

arr = np.array([[1.0, 2.0], [5.0, 6.0]])

arr2 = np.array([[4.0, 5.0], [1.0, 2.0]])

# print(np.dot(arr, arr2))


class network:
    def __init__(self):
        self.w = {}

        self.w["W1"] = np.array([[1.0, -4.0, 2.0], [2.0, 6.0, 3.0]])  # (2,3)
        self.w["B1"] = np.array([-5.0, -4.0, 3.0])  # (3,)

        self.w["W2"] = np.array([[-1.0, 2.0], [3.0, 2.0], [-1.0, -2.0]])  # (3,2)
        self.w["B2"] = np.array([-1.0, -2.0])  # (2,)

        self.w["W3"] = np.array([[2.0, 6.0], [-4.0, 3.0]])  # (2,2)
        self.w["B3"] = np.array([3.0, -1.0])  # (2,)

    def forward(self, X):

        W1, W2, W3 = self.w["W1"], self.w["W2"], self.w["W3"]
        B1, B2, B3 = self.w["B1"], self.w["B2"], self.w["B3"]

        A1 = B1 + np.dot(X, W1)
        Z1 = functions.sigmoid_func(A1)

        A2 = B2 + np.dot(Z1, W2)
        Z2 = functions.sigmoid_func(A2)

        A3 = B3 + np.dot(Z2, W3)

        Y = functions.softmax_func(A3)

        print(Y)


if __name__ == "__main__":
    n = network()
    n.forward(np.array([1.0, -1.0]))
