import numpy as np
import matplotlib.pyplot as plt


def step_func(x) -> float:
    return 1.0 if x > 0 else 0.0


def sigmoid_func(x) -> float:
    return 1 / (np.exp(-x) + 1)


def identity_func(x) -> float:
    return x


def softmax_func(x) -> float:
    c = np.max(x)
    exp = np.exp(x - c)
    exp_sum = np.sum(exp)
    return exp / exp_sum

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / 2*h

def numerical_gradient(f, x):
    y = x
    print("x[0] shpae ", x.shape[0])
    for i in range(x.shape[0]):
        h = 1e-4
        print(" x : ", x)
        arr1 = arr2 = x
        arr1[i] += h
        arr2[i] -= h
        print("arr1 = ", arr1)
        print("arr2 = ", arr2)
        y[i] = (f(arr1) - f(arr2)) / 2*h

    return y
        