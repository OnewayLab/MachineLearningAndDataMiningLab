import time
import torch
import numpy as np


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def accuracy(self, x, y):
        return torch.sum(self.predict(x) == y).item() / len(y)

    def loss(self, x, y):
        x = self.forward(x)
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
        return torch.nn.functional.cross_entropy(x, y)


class CNNClassifier(torch.nn.Module):
    def __init__(self, output_dim):
        super(CNNClassifier, self).__init__()
        self.conv_layers = torch.nn.ModuleList(
            [
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            torch.nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ]
        )
        self.fc_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(16 * 5 * 5, 120),
                torch.nn.ReLU(),
                torch.nn.Linear(120, 84),
                torch.nn.ReLU(),
                torch.nn.Linear(84, output_dim),
            ]
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        for layer in self.conv_layers:
            x = layer(x)
        #     print(x.shape)
        # exit()
        x = x.view(-1, 16 * 5 * 5)
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def accuracy(self, x, y):
        return torch.sum(self.predict(x) == y).item() / len(y)

    def loss(self, x, y):
        x = self.forward(x)
        return torch.nn.functional.cross_entropy(x, y)

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.layers = torch.nn.Sequential (
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.nn.functional.relu(self.layers(x) + self.shortcut(x))


class ResNet18(torch.nn.Module):
    def __init__(self, output_dim):
        super(ResNet18, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.fc = torch.nn.Linear(512, output_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def accuracy(self, x, y):
        return torch.sum(self.predict(x) == y).item() / len(y)

    def loss(self, x, y):
        x = self.forward(x)
        return torch.nn.functional.cross_entropy(x, y)
