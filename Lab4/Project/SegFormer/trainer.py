import torch
import numpy as np
import os
import sys
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from transformers import get_scheduler

from utils.metrics import calculate_metric_percase

def train(model, train_ds, device, batch_size, accumulate_steps, epochs, patience, lr, model_path, output_path):
    """
    训练模型
    :param model: 模型
    :param train_ds: 训练集
    :param device: 设备
    :param batch_size: 批大小
    :param accumulate_steps: 梯度累积步数
    :param epochs: 迭代次数
    :param patience: 提前停止的容忍度，连续 patience 次的验证集损失没有下降则停止训练
    :param lr: 学习率
    :param output_path: 模型保存路径
    :param output_path: 日志和图像保存路径
    """
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train")
    handler = logging.FileHandler(os.path.join(output_path, "train.log"), "w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # 打印超参数
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}, Patience: {patience}, Learning rate: {lr}")

    # 处理数据集
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
    train_size, val_size = len(train_ds), len(val_ds)
    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=int(batch_size/2), pin_memory=True
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=int(batch_size/2), pin_memory=True
    )
    logger.info(f"Train set size: {train_size}")
    logger.info(f"Validation set size: {val_size}")

    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs * len(train_dataloader),
    )

    model.to(device)

    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    patience_count = 0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        # 训练
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        for i, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / accumulate_steps
            loss.backward()
            total_loss += loss.item() * len(batch)
            if (i + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
        train_loss.append(total_loss / train_size)
        # 验证
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in tqdm(val_dataloader, total=len(val_dataloader), desc="Validating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * len(batch)
            val_loss.append(total_loss / val_size)
        # 记录损失
        logger.info(f"\tTraining loss: {train_loss[-1]}")
        logger.info(f"\tValidation loss: {val_loss[-1]}")
        # 保存最好的模型
        if total_loss < best_val_loss:
            best_val_loss = total_loss
            torch.save(model.state_dict(), os.path.join(model_path, "best.pth"))
            patience_count = 0
            logger.info("\tBest model saved!")
        else:
            patience_count += 1
            if patience_count == patience:
                logger.info("\tEarly stopping!")
                break

    # 绘制损失曲线
    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_path, "loss.png"))


def test(model, test_ds, device="cpu", batch_size=48, output_path="./output"):
    """
    测试模型
    :param model: 模型
    :param test_ds: 测试集
    :param device: 设备
    :param batch_size: 批大小
    :param output_path: 输出路径
    """
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("test")
    handler = logging.FileHandler(os.path.join(output_path, "test.log"), "w")
    handler.setLevel(logging.INFO)
    # handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"Test set size: {len(test_ds)}")

    model.to(device)

    # 测试
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for case in tqdm(test_ds, total=len(test_ds)):
            image, label = case["pixel_values"], case["labels"]
            image = image[:, np.newaxis, :, :]
            prediction = np.zeros_like(label)
            for i in range(0, len(image), batch_size):
                batch_x = image[i : i + batch_size]
                batch_x = torch.from_numpy(batch_x).to(device)
                logits = model(batch_x).logits
                logits = interpolate(
                    logits,
                    size=label.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                prediction[i : i + batch_size] = logits.argmax(dim=1).cpu().numpy()
            metric_i = []
            for i in range(1, model.config.num_labels):
                metric_i.append(calculate_metric_percase(prediction == i, label == i))
            metric_list += np.array(metric_i)
    metric_list /= len(test_ds)
    for i in range(1, model.config.num_labels):
        logger.info(f"Class {i} {model.config.id2label[i]} mean_dice: {metric_list[i - 1][0]}, mean_hd95: {metric_list[i - 1][1]}")
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logger.info(f"Testing performance in best val model: mean_dice: {performance} mean_hd95: {mean_hd95}")
