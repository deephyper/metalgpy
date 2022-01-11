import metalgpy as mpy
import numpy as np
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


@mpy.meta
class NeuralNetwork(nn.Module):
    def __init__(self, u1, u2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, u1),
            nn.ReLU(),
            nn.Linear(u1, u2),
            nn.ReLU(),
            nn.Linear(u2, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Explorable model
model = NeuralNetwork(u1=mpy.Int(16, 32), u2=mpy.Int(16, 32)).to(device)
print("Explorable model: ", model)

rng = np.random.RandomState(42)

for _, model in mpy.sample(model, size=5, rng=rng):
    print("\n * sample new random model ->", model)
    model = model.evaluate() # instanciate torch model
    print(model)
