import sys, os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
arr = x_train[0]
arr = arr.reshape(28, 28)
img = Image.fromarray(np.uint8(arr))
img.show()
