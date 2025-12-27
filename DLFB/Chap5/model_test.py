import numpy as np
import os, sys
from network import network
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
image_path = os.path.join(IMAGE_DIR, "2.png")

sys.path.append(BASE_DIR)
sys.path.append(MODEL_DIR)
sys.path.append(IMAGE_DIR)

from dataset.mnist import load_mnist

img = Image.open(image_path)
img = img.convert("L")

image_arr = np.array(img)
image_arr = image_arr.astype(float) / 255.0

data = np.load("params.npz")
print(type(data))
params = {k: data[k] for k in data.files}

n = network(784, 50, 10)
n.initParams(params)

x = image_arr.reshape(1, 784)
t = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
print(n.answer(x, t))


img.show()
