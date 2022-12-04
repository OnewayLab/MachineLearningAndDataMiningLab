import shutil
import numpy as np
import matplotlib.pyplot as plt
import os


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centers = None

    def fit(self, X):
        """
        训练模型
        :param X: 训练集
        """
        # 随机选择K个点作为初始质心
        mask = np.random.choice(X.shape[0], self.k)
        self.centers = X[mask]

        while True:
            # 将每个点指派到最近的质心
            category = [[] for _ in range(self.k)]
            for point in X:
                dist = self.distance(point, self.centers)
                point_category = np.argmin(dist)
                category[point_category].append(point)

            # 计算新质心坐标
            new_centers = np.zeros_like(self.centers)
            for j in range(self.k):
                new_centers[j] = np.sum(np.array(category[j]), axis=0) / len(
                    category[j]
                )

            # 质心无变化则结束循环
            if (new_centers == self.centers).all():
                break

            # 更新质心坐标
            self.centers = new_centers

    def predict(self, X):
        """
        预测
        :param X: 测试集
        :return: 预测结果，每个点对应的聚类类别，用数字 1-K 表示
        """
        predict = [np.argmin(self.distance(point, self.centers)) for point in X]
        return np.array(predict, dtype=int)

    def distance(self, point, center):
        """
        计算一个点到质心的距离
        :param point: 一个点的坐标
        :param center: 质心坐标构成的矩阵
        :return: 数据集每个点到各个质心的距离构成的矩阵
        """
        # 欧氏距离
        dist = np.sqrt(np.sum((point - center) ** 2, axis=1))
        return dist
