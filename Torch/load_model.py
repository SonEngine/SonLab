import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os, sys
from save_model import model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/mlp2.pt")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "models/2.png")


img = Image.open(IMAGE_DIR).convert("L")  # L: grayscale
img = img.resize((28, 28))
arr = np.array(img, dtype=np.float32)
arr /= 255.0
arr = arr.reshape(1, -1)

device = torch.device("cpu")
x = torch.from_numpy(arr).to(device)
model = torch.jit.load(MODEL_DIR, map_location=device)
model.eval()

with torch.no_grad():
    y = model(x)

probs = torch.softmax(y, dim=1)
print("예측 클래스:", torch.argmax(probs, dim=1).item())
