import numpy as np
import sys, os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

sys.path.append(BASE_DIR)
sys.path.append(MODEL_DIR)
save_path = os.path.join(MODEL_DIR, "test2.npz")
params = {"w0": np.array([[0, 1, 2], [3, 4, 5]])}


np.savez_compressed(save_path, **params)
print("saved to:", save_path)
