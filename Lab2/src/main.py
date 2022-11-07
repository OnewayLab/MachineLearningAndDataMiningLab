import torch
import numpy as np
import os

from classifiers import LinearClassifier, MLPClassifier, CNNClassifier
from trainer import trainer
from load_data import load_data

# 定义路径
MODEL_PATH = "../model/"
FIGURE_PATH = "../figures/"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

# 输入输出大小
INPUT_SIZE = 32 * 32 * 3
OUTPUT_SIZE = 10


if __name__ == "__main__":
    # 读取数据
    X_train, Y_train, X_test, Y_test = load_data()

    # 数据预处理
    X_train = torch.from_numpy(X_train).float() / 255 * 2 - 1
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(X_test).float() / 255 * 2 - 1
    Y_test = torch.from_numpy(Y_test).long()

    # 训练模型
    model = LinearClassifier(INPUT_SIZE, OUTPUT_SIZE)
    trainer(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr=1e-1,
        optimizer="SGD",
        model_path="model/BestLinearClassfier.pt",
        plot_path="figures/LinearClassfier.png",
    )
