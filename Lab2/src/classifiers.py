import time
import torch
import numpy as np


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
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

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, num_hidden=1):
        super(MLPClassifier, self).__init__()
        self.num_hidden = num_hidden
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for i in range(num_hidden - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.output_functin = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for i in range(self.num_hidden):
            x = self.layers[i](x)
            x = torch.nn.functional.relu(x)
        x = self.layers[self.num_hidden](x)
        return x

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def accuracy(self, x, y):
        return torch.sum(self.predict(x) == y).item() / len(y)

    def loss(self, x, y):
        x = self.forward(x)
        x = self.output_functin(x)
        return torch.nn.functional.cross_entropy(x, y)

class CNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, num_hidden=1):
        super(CNNClassifier, self).__init__()
        self.num_hidden = num_hidden
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Conv2d(1, hidden_dim, kernel_size=5, stride=1, padding=2))
        self.layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        for i in range(num_hidden - 1):
            self.layers.append(torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2))
            self.layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layers.append(torch.nn.Linear(hidden_dim * 7 * 7, output_dim))
        self.output_functin = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        for i in range(self.num_hidden):
            x = self.layers[2 * i](x)
            x = torch.nn.functional.relu(x)
            x = self.layers[2 * i + 1](x)
        x = x.view(-1, 7 * 7 * 100)
        x = self.layers[2 * self.num_hidden](x)
        return x

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def accuracy(self, x, y):
        return torch.sum(self.predict(x) == y).item() / len(y)

    def loss(self, x, y):
        x = self.forward(x)
        x = self.output_functin(x)
        return torch.nn.functional.cross_entropy(x, y)