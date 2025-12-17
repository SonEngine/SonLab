import functions
import numpy as np

def func(x):
    return x*x + 3*x

def func2(x):
    return x[0]**2 + x[1]**2


if __name__ == "__main__":
    
    arr = np.array([3.,4.])
    print(functions.numerical_gradient(func2, arr))
    