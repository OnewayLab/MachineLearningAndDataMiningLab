import time
import torch
import numpy as np


class LinearClassfier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassfier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.output_functin = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def accuracy(self, x, y):
        return torch.sum(self.predict(x) == y).item() / len(y)

    def loss(self, x, y):
        x = self.forward(x)
        x = self.output_functin(x)
        return torch.nn.functional.cross_entropy(x, y)