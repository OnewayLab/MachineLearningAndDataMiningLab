import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def trainer(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=32,
    epochs=1000,
    pacient=4,
    lr=1e-3,
    optimizer="SGD",
    model_path="best_model.pt",
    plot_path="loss.png"
):
    """
    训练模型
    :param model: 模型
    :param x_train: 训练集输入
    :param y_train: 训练集标签
    :param x_test: 测试集输入
    :param y_test: 测试集标签
    :param batch_size: 批大小
    :param epochs: 迭代次数
    :param pacient: 提前停止的容忍度，连续 pacient 次的验证集损失没有下降则停止训练
    :param lr: 学习率
    :param optimizer: 优化器，可选择 "SGD"、"Momentum" 或 "Adam"
    :param model_path: 模型保存路径
    :param plot_path: 图像保存路径
    """
    # 打印超参数
    print(f"Model: {type(model).__name__}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, Pacient: {pacient}, Learning rate: {lr}, Optimizer: {optimizer}")

    # 打乱输入
    print("Processing Data")
    index = np.arange(len(x_train))
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    # 切分训练集和验证集
    train_x = x_train[: int(len(x_train) * 0.8)]
    train_y = y_train[: int(len(y_train) * 0.8)]
    val_x = x_train[int(len(x_train) * 0.8) :]
    val_y = y_train[int(len(y_train) * 0.8) :]

    # 选择优化器
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer must be one of SGD, Momentum or Adam")

    start_time = time.time()
    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    pacient_count = 0
    print("Start Training")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # 训练
        total_loss = 0
        for i in tqdm(range(0, len(train_x), batch_size)):
            batch_x = train_x[i : i + batch_size]
            batch_y = train_y[i : i + batch_size]
            optimizer.zero_grad()
            loss = model.loss(batch_x, batch_y)
            total_loss += loss.item() * len(batch_x)
            loss.backward()
            optimizer.step()
        train_loss.append(total_loss / len(train_x))
        print(f"\tTraining loss: {train_loss[-1]}")
        # 验证
        with torch.no_grad():
            total_loss = 0
            for i in range(0, len(val_x), batch_size):
                batch_x = val_x[i : i + batch_size]
                batch_y = val_y[i : i + batch_size]
                loss = model.loss(batch_x, batch_y)
                total_loss += loss.item() * len(batch_x)
            val_loss.append(total_loss / len(val_x))
            print(f"\tValidation loss: {val_loss[-1]}")
            # 保存最好的模型
            if total_loss < best_val_loss:
                best_val_loss = total_loss
                torch.save(model.state_dict(), model_path)
                pacient_count = 0
                print("\tBest model saved!")
            else:
                pacient_count += 1
                if pacient_count == pacient:
                    print("\tEarly stopping!")
                    break
    print(f"Training finished in {time.time() - start_time}s")

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_path)

    with torch.no_grad():
        # 加载最好的模型
        model.load_state_dict(torch.load(model_path))
        print("Best model loaded!")

        # 测试
        print("Start Testing")
        model.eval()
        test_acc = 0
        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i : i + batch_size]
            batch_y = y_test[i : i + batch_size]
            test_acc += (model(batch_x).argmax(dim=1) == batch_y).sum().item()
        test_acc /= len(x_test)
        print(f"Test set accuracy: {test_acc * 100}%")