import functions
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x * x + 3 * x


def func2(x):
    return x[0] ** 2 + x[1] ** 2


def test_gradient():
    x = np.arange(-2, 2.5, 0.25)
    y = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x, y)
    # print(X)
    points = np.column_stack([X.ravel(), Y.ravel()])
    grad = functions.numerical_gradient(func2, points)

    U = grad[:, 0]
    W = grad[:, 1]

    xmin = ymin = -2
    xmax = ymax = 2

    plt.axis([xmin, xmax, ymin, ymax])
    plt.axhline(0)
    plt.axvline(0)
    plt.quiver(X, Y, U, W, angles="xy", scale_units="xy", scale=10)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.show()


def test_gradient_descent():
    x = np.array([-3.0, -3.0])
    ret = np.stack(functions.gradient_descent(func2, x, 0.1))
    plt.scatter(ret[:, 0], ret[:, 1])
    plt.show()


if __name__ == "__main__":

    test_gradient_descent()
