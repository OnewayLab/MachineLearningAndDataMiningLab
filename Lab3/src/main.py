import time
import numpy as np
import os
from sklearn import metrics
from kmeans import KMeans
from scipy.optimize import linear_sum_assignment

# 数据集路径
TRAIN_PATH = "../data/mnist_train.csv"
TEST_PATH = "../data/mnist_test.csv"

# 聚类个数
K = 10

def evaluate(predict, label):
    """
    评估聚类结果
    :param predict: 预测结果
    :param label: 标签
    :return: 聚类准确率
    """
    predict = predict.astype(np.int64)
    label = label.astype(np.int64)
    assert predict.size == label.size
    D = int(max(predict.max(), label.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(predict.size):
        w[predict[i], label[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / predict.size


if __name__ == "__main__":
    # 读取数据集
    print("Loading data...")
    train_data = np.loadtxt(open(TRAIN_PATH, "rb"), delimiter=",", skiprows=1)
    test_data = np.loadtxt(open(TEST_PATH, "rb"), delimiter=",", skiprows=1)
    np.random.shuffle(train_data)
    train_X, train_y = train_data[:, 1:], train_data[:, 0]
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    # KMeans聚类
    print("Training K-Means model...")
    start_time = time.time()
    kmeans = KMeans(K)
    kmeans.fit(train_X)
    train_predict = kmeans.predict(train_X)
    test_predict = kmeans.predict(test_X)
    end_time = time.time()
    print("K-Means model training time: {}s".format(end_time - start_time))

    # 评估聚类结果
    print("Evaluating K-Means model...")
    train_score = evaluate(train_y, train_predict)
    test_score = evaluate(test_y, test_predict)

    print("Train score: {}".format(train_score))
    print("Test score: {}".format(test_score))