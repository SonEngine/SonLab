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
    return (f(x + h) - f(x - h)) / 2 * h


def numerical_gradient(f, x: np.ndarray) -> np.ndarray:
    grad = x.copy()
    if x.ndim == 1:
        for i in range(x.shape[0]):
            h = 1e-4
            x0 = x.copy()
            x1 = x.copy()
            x0[i] += h
            x1[i] -= h
            grad[i] = (f(x0) - f(x1)) / (2 * h)
    else:
        for i in range(x.shape[0]):
            h = 1e-4
            for j in range(x.shape[1]):
                x0 = x[i].copy()
                x1 = x[i].copy()
                x0[j] += h
                x1[j] -= h
                grad[i][j] = (f(x0) - f(x1)) / (2 * h)
    return grad


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshpae(1, t.size)

    batch_size = y.shape[0]
    epsilon = 1e-7
    return -np.sum(t * np.log(y + epsilon)) / batch_size


def gradient_descent(f, x: np.ndarray, eta: float = 0.1):
    i = 0
    l = [x]
    while True:
        # print(f"{i}번째 x : {x}")
        grad = numerical_gradient(f, x)
        abs_grad = np.abs(grad)
        sum = np.sum(abs_grad)
        if sum < 1e-3:
            break
        x = x - grad * eta
        l.append(x)
        i += 1

    return l
