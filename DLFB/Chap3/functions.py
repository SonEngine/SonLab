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
