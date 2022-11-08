import torch
import numpy as np
import os

from classifiers import LinearClassifier, MLPClassifier, CNNClassifier, ResNet18
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

# 定义训练设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 读取数据
    X_train, Y_train, X_test, Y_test = load_data()

    # 数据预处理
    X_train = torch.from_numpy(X_train).float() / 255 * 2 - 1
    Y_train = torch.from_numpy(Y_train).long()
    X_test = torch.from_numpy(X_test).float() / 255 * 2 - 1
    Y_test = torch.from_numpy(Y_test).long()

    X_train = X_train.to(DEVICE)
    Y_train = Y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    Y_test = Y_test.to(DEVICE)
    # # 线性分类器
    # linear_classifier = LinearClassifier(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
    # trainer(
    #     linear_classifier,
    #     X_train,
    #     Y_train,
    #     X_test,
    #     Y_test,
    #     lr=0.0001,
    #     optimizer="Adam",
    #     model_path=MODEL_PATH + "BestLinearClassifier.pt",
    #     plot_path=FIGURE_PATH + "LinearClassifier.png",
    # )

    # # 多层感知机
    # mlp_classifier = MLPClassifier(
    #     INPUT_SIZE, OUTPUT_SIZE, # num_hidden=4, hidden_dim=600
    # ).to(DEVICE)
    # trainer(
    #     mlp_classifier,
    #     X_train,
    #     Y_train,
    #     X_test,
    #     Y_test,
    #     lr=0.0001,
    #     optimizer="Adam",
    #     model_path=MODEL_PATH + "BestMLPClassifier.pt",
    #     plot_path=FIGURE_PATH + "MLPClassifier.png",
    # )

    # # 卷积神经网络
    # cnn_classifier = CNNClassifier(OUTPUT_SIZE).to(DEVICE)
    # trainer(
    #     cnn_classifier,
    #     X_train,
    #     Y_train,
    #     X_test,
    #     Y_test,
    #     lr=0.0001,
    #     optimizer="Adam",
    #     model_path=MODEL_PATH + "BestCNNClassifierAdamNoPool.pt",
    #     plot_path=FIGURE_PATH + "CNNClassifierAdamNoPool.png",
    # )

    # Res-18
    cnn_classifier = ResNet18(OUTPUT_SIZE).to(DEVICE)
    trainer(
        cnn_classifier,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr=0.001,
        optimizer="Adam",
        model_path=MODEL_PATH + "BestResNetClassifierAdam.pt",
        plot_path=FIGURE_PATH + "ResNetClassifierAdam.png",
    )
