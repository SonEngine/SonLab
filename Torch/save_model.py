import torch
import torch.nn as nn
import numpy as np
import sys, os

save_path = os.path.join(os.path.dirname(__file__), "models/mlp2.pt")
npz_path = os.path.join(os.path.dirname(__file__), "models/params.npz")


class model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.device = torch.device("cpu")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def load_weights(self, npz_path: str):
        data = np.load(npz_path)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(data["w1"].T).float())
            self.fc1.bias.copy_(torch.from_numpy(data["b1"].T).float())
            self.fc2.weight.copy_(torch.from_numpy(data["w2"].T).float())
            self.fc2.bias.copy_(torch.from_numpy(data["b2"].T).float())

    def save(self, save_path):
        self.eval()
        ex = torch.randn(1, 784)
        ts = torch.jit.trace(self, ex)
        ts.save(save_path)


m = model(784, 50, 10)
m.load_weights(npz_path)
m.save(save_path)
