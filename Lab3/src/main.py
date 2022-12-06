import time
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kmeans import KMeans
from gmm import GMM

np.random.seed(5678)

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
    train_data, validate_data = train_data[:int(len(train_data) * 0.8)], train_data[int(len(train_data) * 0.8):]
    test_data = np.loadtxt(open(TEST_PATH, "rb"), delimiter=",", skiprows=1)
    np.random.shuffle(train_data)
    train_X, _ = train_data[:, 1:], train_data[:, 0]
    validate_X, validate_y = validate_data[:, 1:], validate_data[:, 0]
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    # 使用 PCA 进行降维
    pca = PCA(50)
    train_X = pca.fit_transform(train_X)
    validate_X = pca.transform(validate_X)
    test_X = pca.transform(test_X)

    # 定义模型
    models = {
        "K-Means (random init)": KMeans(K, init_method="random"),
        "K-Means (distance-based init)": KMeans(K, init_method="distance"),
        # "GMM (random init, ordinary cov)":GMM(K, init_method="random", cov_type="ordinary"),
        # "GMM (random init, diag cov)":GMM(K, init_method="random", cov_type="diag"),
        # "GMM (random init, diag cov with the same element": GMM(K, init_method="random", cov_type="diag_same"),
        # "GMM (random posterior init, ordinary cov": GMM(K, init_method="random_posterior", cov_type="ordinary"),
        # "GMM (init by k-means, ordinary cov": GMM(K, init_method="k-means", cov_type="ordinary"),
    }

    # 各个模型训练时验证集准去率
    validate_acc = {k: [] for k in models}

    for name, model in models.items():
        # 训练模型
        print(f"Training {name}...")
        start_time = time.time()
        model.train(
            train_X,
            lambda: validate_acc[name].append(evaluate(model.predict(validate_X), validate_y))
        )
        end_time = time.time()
        print("K-Means model training time: {}s".format(end_time - start_time))
        # 评估聚类结果
        print("Evaluating K-Means model...")
        test_predict = model.predict(test_X)
        test_score = evaluate(test_y, test_predict)
        print("Test score: {}".format(test_score))

    # 画曲线
    for name, acc_list in validate_acc.items():
        plt.plot(acc_list, label=name)
    plt.legend()
    plt.show()