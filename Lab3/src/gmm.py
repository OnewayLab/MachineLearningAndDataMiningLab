import numpy as np
from scipy.stats import multivariate_normal
from kmeans import KMeans


class GMM:
    def __init__(self, k, init_method="random", cov_type="ordinary"):
        """
        高斯混合模型
        :param k: 聚类数目
        :param init_method: 初始化方法，可选“random”、“random_posterior”和“k-means”
        :param cov_type: 协方差类型，可选“ordinary”（普通矩阵）、“diag”（对角阵）和“diag_same”（元素值相等的对角阵）
        """
        self.k = k
        self.init_method = init_method
        self.cov_type = cov_type
        self.means = None
        self.covs = None
        self.priors = None

    def train(self, X, max_iter=100, call_back=None):
        """
        训练模型
        :param X: 训练集
        :param max_iter: 最大迭代次数
        :param call_back: 回调函数，在每次迭代结束时调用
        """
        # 初始化参数
        if self.init_method == "random":
            # 随机抽取 K 个点作为初始均值
            indices = np.random.choice(X.shape[0], self.k)
            self.means = X[indices]
            # 使用样本数据的协方差矩阵作为初始协方差
            cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1]) # 防止协方差矩阵为奇异矩阵
            if self.cov_type == "diag":
                cov = np.diag(np.diag(cov))
            elif self.cov_type == "diag_same":
                cov = np.diag([np.diag(cov).mean()] * X.shape[1])
            self.covs = cov[np.newaxis, :].repeat(self.k, axis=0)
            # 初始化先验概率
            self.priors = np.random.rand(self.k)
            self.priors /= np.sum(self.priors)  # 保证和为 1
        elif self.init_method == "random_posterior":
            # 随机初始化后验概率
            posterior = np.random.rand(X.shape[0], self.k)
            posterior /= np.sum(posterior, axis=1, keepdims=True)
            # 用 M 步计算各个参数
            self.means, self.covs, self.priors = self.__M_step(X, posterior)
        elif self.init_method == "k-means":
            # 使用K-means初始化参数
            kmeans = KMeans(self.k)
            kmeans.train(X)
            predict = kmeans.predict(X)
            self.means = kmeans.centers
            self.covs = np.zeros((self.k, X.shape[1], X.shape[1]))
            self.priors = np.zeros(self.k)
            for j in range(self.k):
                self.covs[j] = np.cov(X[predict == j], rowvar=False)
                self.priors[j] = np.sum(predict == j) / X.shape[0]
        else:
            raise ValueError("Invalid init_method!")

        # 训练
        for i in range(max_iter):
            print(f"Iter: {i}")
            # E步，计算隐变量的后验概率
            posterior = self.__E_step(X)
            # M步，更新参数
            new_means, new_covs, new_priors = self.__M_step(X, posterior)
            # 参数无变化则结束循环
            if (new_means == self.means).all() and (new_covs == self.covs).all() and (
                new_priors == self.priors
            ).all():
                break
            # 更新参数
            self.means = new_means
            self.covs = new_covs
            self.priors = new_priors
            #调用回调函数
            if call_back:
                call_back()

    def __E_step(self, X):
        """
        E步，计算隐变量的后验概率
        :param X: 训练集
        :return: 隐变量的后验概率
        """
        posterior = np.zeros((X.shape[0], self.k))
        for j in range(self.k):
            posterior[:, j] = self.priors[j] * self.__gaussian(X, self.means[j], self.covs[j])
        posterior /= np.sum(posterior, axis=1, keepdims=True)
        return posterior

    def __M_step(self, X, posterior):
        """
        M步，更新参数
        :param X: 训练集
        :param posterior: 隐变量的后验概率
        :return: 更新后的参数
        """
        Nk = np.sum(posterior, axis=0)

        # 更新均值
        new_means = np.dot(posterior.T, X) / Nk[:, np.newaxis]

        # 更新协方差
        new_covs = np.zeros((self.k, X.shape[1], X.shape[1]))
        for j in range(self.k):
            diff = X - new_means[j]
            new_cov = np.dot(posterior[:, j] * diff.T, diff) / (Nk[j] + 1e-6) # 防止除零
            # new_cov += np.eye(new_cov.shape[0]) # 防止为奇异矩阵
            if self.cov_type == "diag":
                new_cov = np.diag(np.diag(new_cov))
            elif self.cov_type == "diag_same":
                new_cov = np.diag([np.diag(new_cov).mean()] * X.shape[1])
            new_covs[j] = new_cov

        # 更新先验概率
        new_priors = Nk / X.shape[0]
        return new_means, new_covs, new_priors

    def __gaussian(self, X, mean, cov):
        """
        计算高斯分布的概率密度
        :param X: 训练集
        :param mean: 均值
        :param cov: 协方差
        :return: 高斯分布的概率密度
        """
        var = multivariate_normal(mean=mean, cov=cov)
        return var.pdf(X)

    def predict(self, X):
        """
        预测
        :param X: 测试集
        :return: 预测结果
        """
        # 计算高斯分布的概率密度
        posterior = np.zeros((X.shape[0], self.k))
        for j in range(self.k):
            posterior[:, j] = self.priors[j] * self.__gaussian(X, self.means[j], self.covs[j])

        # 计算隐变量的后验概率
        posterior /= np.sum(posterior, axis=1, keepdims=True)

        # 返回后验概率最大的类别
        return np.argmax(posterior, axis=1)
