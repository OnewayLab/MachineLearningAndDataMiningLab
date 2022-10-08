import numpy as np
import pandas as pd
from sklearn import svm

# 文件路径
TRAIN_PATH = "data/mnist_01_train.csv"
TEST_PATH = "data/mnist_01_test.csv"


def data_process(
    train_path: str, test_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    数据处理
    :param train_path: 训练集路径
    :param test_path: 测试集路径
    :return: 训练集输入、训练集标签、测试集输入、测试集标签
    """
    # 读取数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values

    # 归一化
    X_min = np.min(X_train, axis=0)
    X_max = np.max(X_train, axis=0) + 1e-7
    X_train = (X_train - X_min) / X_max
    X_test = (X_test - X_min) / X_max

    # 把 {0, 1} 标签转换为 {-1, 1} 标签
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    return X_train, y_train, X_test, y_test


class LinearClassifier:
    def __init__(
        self,
        loss="hinge",
        learning_rate=0.01,
        max_iter=1000,
        batch_size=32,
    ):
        """
        线性分类器
        :param loss: 损失函数类型，可选 "hinge" 或 "cross_entropy"
        :param learning_rate: 学习率
        :param max_iter: 最大迭代次数
        :param batch_size: batch 大小
        """
        if loss == "hinge":
            self.loss = lambda X, y: np.mean(
                np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b))
            )
            self.loss_gradient = lambda X, y: (
                np.dot(X.T, (y * (np.dot(X, self.w) + self.b) < 1).astype(int) * -y)
                / X.shape[0],
                np.mean((y * (np.dot(X, self.w) + self.b) < 1).astype(int) * -y),
            )
        elif loss == "cross_entropy":
            self.loss = lambda X, y: np.mean(
                np.log(1 + np.exp(-y * (np.dot(X, self.w) + self.b)))
            )
            self.loss_gradient = lambda X, y: (
                -np.dot(X.T, y / (1 + np.exp(y * (np.dot(X, self.w) + self.b))))
                / X.shape[0],
                -np.mean(y / (1 + np.exp(y * (np.dot(X, self.w) + self.b)))),
            )
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型
        :param X: 输入
        :param y: 标签
        :return: None
        """
        # 初始化权重和偏置
        self.w = np.random.normal(size=X.shape[1])
        self.b = np.random.normal(size=1)

        # 训练
        for i in range(self.max_iter):
            # 随机抽取 batch
            batch_idx = np.random.choice(X.shape[0], self.batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # 计算梯度
            dw, db = self.loss_gradient(X_batch, y_batch)

            # 更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # 打印训练过程
            if i % 100 == 0:
                print("iter: {}, loss: {}".format(i, self.loss(X, y)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        :param X: 输入
        :return: 预测结果
        """
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        :param X: 输入
        :param y: 标签
        :return: 准确率
        """
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # 数据处理
    X_train, y_train, X_test, y_test = data_process(TRAIN_PATH, TEST_PATH)

    # 使用 hinge loss 的线性分类器
    print("使用 hinge loss 训练")
    hinge_loss_model = LinearClassifier(
        loss="hinge",
        learning_rate=0.1,
        max_iter=2000,
        batch_size=32,
    )
    hinge_loss_model.fit(X_train, y_train)
    train_acc = hinge_loss_model.score(X_train, y_train)
    test_acc = hinge_loss_model.score(X_test, y_test)
    print("训练集准确率: {}, 测试集准确率: {}".format(train_acc, test_acc))

    # 使用交叉熵误差的线性分类器
    print("使用交叉熵损失训练")
    cross_entropy_loss_model = LinearClassifier(
        loss="cross_entropy",
        learning_rate=0.1,
        max_iter=2000,
        batch_size=32,
    )
    cross_entropy_loss_model.fit(X_train, y_train)
    train_acc = cross_entropy_loss_model.score(X_train, y_train)
    test_acc = cross_entropy_loss_model.score(X_test, y_test)
    print("训练集准确率: {}, 测试集准确率: {}".format(train_acc, test_acc))

    # 使用线性核函数的 SVM
    print("使用线性核函数的 SVM")
    linear_svm_model = svm.SVC(C=1, kernel="linear", gamma="auto")
    linear_svm_model.fit(X_train, y_train)
    print("训练集准确率: {}, 测试集准确率: {}".format(linear_svm_model.score(X_train, y_train), linear_svm_model.score(X_test, y_test)))

    # 使用高斯核函数的 SVM
    print("使用高斯核函数的 SVM")
    rbf_svm_model = svm.SVC(C=1, kernel="rbf", gamma="auto")
    rbf_svm_model.fit(X_train, y_train)
    print("训练集准确率: {}, 测试集准确率: {}".format(rbf_svm_model.score(X_train, y_train), rbf_svm_model.score(X_test, y_test)))
