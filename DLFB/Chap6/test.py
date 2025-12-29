import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/ (np.exp(-x) + 1)
x = np.random.randn(1000,100)
activations = {}
h_size = 5

for i in range(h_size):
    if i!=0:
        x = activations[i-1]
    if i==0:
        w = np.random.randn(100,100) / np.sqrt(100)
    else:
        w = np.random.randn(100,100) / np.sqrt(100)
    a = np.dot(x,w)
    a = sigmoid(a)
    activations[i] = a

for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(),30, range=(0,1))
plt.show()