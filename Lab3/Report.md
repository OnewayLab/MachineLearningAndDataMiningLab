# 机器学习与数据挖掘作业 3

## 1 数据预处理

* 本次实验数据为 MINIST 手写识别数据集，数据集分为训练集和测试集两部分，分别包含 60000 和 10000 个样本。
* 为了能够直观地观察模型性能在训练过程中的变化，从训练集中划分出 20% 的样本作为验证集，剩下 80% 作为训练集。
* 数据集中每个样本为一个 784 维的向量，实验过程中发现，由于样本向量非常稀疏，在计算协方差时很容易由于数值计算上的误差使得计算出的协方差矩阵为奇异矩阵，无法用它进行高斯概率密度函数的计算，而使得 GMM 的训练无法进行。所以，在数据预处理时先使用 PCA 将数据降到 50 维。

## 1 K-Means

### 1.1 算法流程

1. 随机指定 $K$ 个中心 $\boldsymbol{\mu}_k,k=1,\cdots,K$；
2. 把数据点 $\boldsymbol{x}^{(n)}$ 分配到离它最近的中心：
   $$
   r_{nk}=\begin{cases}
   1,&k=\arg\min_j\|\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_j\|^2\\
   0,&其他.
   \end{cases}
   $$
3. 使用每个簇的数据点的平均值作为它的新的中心：
   $$
   \boldsymbol{\mu}_{k} \leftarrow \frac{\sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}}{\sum_{n=1}^{N} r_{n k}}
   $$
4. 重复 2、3 两步，直到聚类中心不再改变。

### 1.2 初始化方法

* 随机初始化

  * 随机选择数据点作为初始中心。
  * 问题：可能选择很近的初始中心。
* 基于距离初始化

  * 从一个随机选择的数据点开始，每次选择离现有聚类中心最远的点。
  * 问题：可能选中离群点。

### 1.3 关键代码

使用一个 `KMeans` 类来表示一个 K-Means 聚类模型，类中有 3 个成员变量：

* `k`：聚类个数
* `centers`：聚类中心
* `init_method`：聚类中心的初始化方式

核心代码是模型的训练函数，这个函数首先根据设定的初始化方式初始化 `k` 个质心，然后根据 [1.1 节](#1.1%20算法流程)中描述的算法进行迭代直到质心不再改变或达到最大迭代次数 `max_iter`。另外，每次迭代结束时会调用回调函数 `callback`，我们可以用它在每次迭代后用验证集验证模型或者打印一些信息。

```python
def train(self, X, max_iter=100, callback=None):
    """
    训练模型
    :param X: 训练集
    :param max_iter: 最大迭代次数
    :param callback: 回调函数，将在每次迭代结束时调用
    """
    # 初始化质心
    if self.init_method == "random":
        # 随机选择K个点作为初始质心
        mask = np.random.choice(X.shape[0], self.k)
    else:
        # 随机选择一个点作为初始质心
        mask = [np.random.choice(X.shape[0])]
        # 选择距离上一个质心最远的点作为下一个质心
        for _ in range(self.k - 1):
            dist = np.array([np.min(self.__distance(point, X[mask])) for point in X])
            mask.append(np.argmax(dist))
    self.centers = X[mask]
    # 训练模型
    for _ in range(max_iter):
        # 将每个点指派到最近的质心
        category = [[] for _ in range(self.k)]
        for point in X:
            dist = self.__distance(point, self.centers)
            point_category = np.argmin(dist)
            category[point_category].append(point)
        # 计算新质心坐标
        new_centers = np.zeros_like(self.centers)
        for j in range(self.k):
            new_centers[j] = np.sum(np.array(category[j]), axis=0) / len(category[j])
        # 质心无变化则结束循环
        if (new_centers == self.centers).all():
            break
        # 更新质心坐标
        self.centers = new_centers
        # 调用回调函数
        if callback:
            callback()
```

### 1.4 实验结果与分析

* 计算聚类精度时需要将模型输出的标签与真实标签使用匈牙利算法进行匹配。
* 为了直观显示测试集上的聚类结果，使用 PCA 将测试集降到 2 维，并将聚类结果用二维散点图表示。

|     初始化方法     |                             随机初始化                             |                                 基于距离初始化                                 |
| :----------------: | :----------------------------------------------------------------: | :----------------------------------------------------------------------------: |
|      训练耗时      |                              80.11 s                              |                                    49.67 s                                    |
| 最终测试集聚类精度 |                              51.58 %                              |                                     55.54%                                     |
|   测试集聚类结果   | ![K-Means 随机初始化训练过程](./figures/K-Means (random init).jpg) | ![K-Means 基于距离初始化训练过程](./figures/K-Means (distance-based init).jpg) |

* 训练过程验证集聚类精度曲线
  <img src=".\figures\K-Means.jpg" alt="训练过程验证集聚类精度曲线" style="zoom:50%;" />
* 测试集真实标签结果
  <img src=".\figures\gold_result.jpg" alt="测试集真实标签结果" style="zoom:50%;" />
* 结果分析
  * 基于距离初始化的 K-Means 模型在测试集上的聚类精度高于随机初始化的模型，从散点图也可看出基于距离初始化的模型最终聚类结果与真实结果更加接近。
  * 两种初始化方式下，训练过程验证集聚类精度都是先升后降，出现了一定程度的过拟合，而随机初始化的模型过拟合比基于距离初始化的模型更加明显。
  * 基于距离初始化的模型需要的迭代次数更少，收敛更快，训练耗时更短。这可能是因为在原始数据的高维空间中，不同类别之间的距离本来就比较大，使用基于距离的初始化方式得到的初始质心很可能位于不同的类别中，更有利于模型的正确收敛；而随机初始化的方式很可能会有两个甚至多个初始质心位于同一类别，要使它们经过迭代移动到不同的类别中就更加困难。
  * 综合以上分析，使用相距较远的样本点作为初始质心的效果要好于随机选择。

## 2 GMM

### 2.1 算法流程

使用期望-最大化算法（EM 算法）训练 GMM 模型的迭代过程可分为两步：

1. E 步：评估期望

   $$
   \mathcal{Q}\left(\boldsymbol{\theta} ; \boldsymbol{\theta}^{(t)}\right)=\sum_{n=1}^{N} \mathbb{E}_{p\left(\boldsymbol{z}_{n} \mid \boldsymbol{x}_{n} ; \boldsymbol{\theta}^{(t)}\right)}\left[\log p\left(\boldsymbol{x}_{n}, \boldsymbol{z}_{n} ; \boldsymbol{\theta}\right)\right]
   $$
2. M 步：更新参数使期望最大化

   $$
   \boldsymbol{\theta}^{(t+1)}=\arg \max _{\boldsymbol{\theta}} \mathcal{Q}\left(\boldsymbol{\theta} ; \boldsymbol{\theta}^{(t)}\right)
   $$

但是，在 M 步中，我们需要的是期望的表达式，而不是期望的具体值，经过公式推导之后，使用 EM 算法训练 GMM 模型的迭代过程可简化为以下两步：

1. 给定当前参数 $\left\{\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}\right\}_{k=1}^{K}$，更新 $\gamma_{nk}$

   $$
   \gamma_{n k} \leftarrow \frac{\mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right) \pi_{k}}{\sum_{i=1}^{K} \mathcal{N}\left(\boldsymbol{x} ; \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \pi_{i}}
   $$
2. 给定 $\gamma_{nk}$，更新 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$ 和 $\pi_k$

   $$
   \begin{array}{c}
   N_{k} \leftarrow \sum_{n=1}^{N} \gamma_{n k} \\
   \boldsymbol{\mu}_{k} \leftarrow \frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} \boldsymbol{x}_{n} \\
   \boldsymbol{\Sigma}_{k} \leftarrow \frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{T} \\
   \pi_{k} \leftarrow \frac{N_{k}}{N}
   \end{array}
   $$

### 2.2 初始化方法

* 随机初始化参数
  * 随机选择 $K$ 个点作为初始均值 $\boldsymbol{\mu}_k$；
  * 使用样本协方差作为每个分布的初始协方差 $\boldsymbol{\Sigma}_k$；
  * 随机初始化先验概率 $\pi_k$，但要注意满足 $\sum_{k=1}^K\pi_k$=1。
* 随机初始化后验概率 $\gamma_{nk}$，然后使用 M 步得到参数 $\boldsymbol{\mu}_k$、$\boldsymbol{\Sigma}_k$ 和 $\pi_k$；
* K-Means
  * 使用 K-Means 算法对训练样本聚类；
  * 使用聚类中心作为每个分布的初始均值 $\boldsymbol{\mu}_k$；
  * 计算每个类内数据的协方差作为每个分布的初始协方差 $\boldsymbol{\Sigma}_k$；
  * 每个类内样本数量除以样本总数得到每个分布的先验概率 $\pi_k$。

### 2.3 协方差矩阵类型

* 普通矩阵
* 对角阵，每个矩阵内元素可以不同
* 对角阵，且每个矩阵内元素相等

### 2.4 关键代码

使用一个 `GMM` 类来表示一个 GMM 聚类模型，类中有 6 个成员变量：

* `k`：聚类个数
* `means`、`covs` 和 `priors`：各个高斯分布的均值、协方差和先验概率
* `init_method`：参数的初始化方式
* `cov_type`：协方差矩阵的类型

核心代码是模型的训练函数，这个函数首先根据设定的初始化方式初始化参数 `means`、`covs` 和 `priors`，然后根据 [2.1 节](#2.1%20算法流程)中描述的算法进行迭代直到参数不再改变或达到最大迭代次数 `max_iter`。另外，每次迭代结束时会调用回调函数 `callback`，我们可以用它在每次迭代后用验证集验证模型或者打印一些信息。

```python
def train(self, X, max_iter=100, callback=None):
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
        if callback:
            callback()
```

### 2.5 实验结果与分析

#### 2.5.1 不同参数初始化方式

|     初始化方法     |                        随机初始化参数                        |                        随机初始化后验                        |                     使用 K-Means 初始化                      |
| :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      训练耗时      |                           77.98 s                            |                           78.14 s                            |                           123.62 s                           |
| 最终测试集聚类精度 |                           60.07 %                            |                           62.30 %                            |                           64.10 %                            |
|   测试集聚类结果   | ![随机初始化参数](.\figures\GMM (random init, ordinary cov).jpg) | ![随机初始化后验](.\figures\GMM (random posterior init, ordinary cov).jpg) | ![使用K-Means初始化](.\figures\GMM (init by k-means, ordinary cov).jpg) |

* 训练过程验证集聚类精度曲线
  <img src=".\figures\ordinary cov.jpg" alt="训练过程验证集聚类精度曲线" style="zoom:50%;" />
* 测试集真实标签结果
  <img src=".\figures\gold_result.jpg" alt="测试集真实标签结果" style="zoom:50%;" />
* 结果分析
  * 使用 K-Means 初始化的方式要好于随机初始化参数和随机初始化后验，随机初始化后验的方式要好于随机初始化参数，从散点图也可看出使用 K-Means 初始化的模型最终聚类结果与真实结果更加接近。这是因为相比于随机选组参数，从 K-Means 聚类结果得到的均值、协方差和先验概率能够更接近最优 GMM 模型的参数。而随机初始化后验概率再通过 M 步得到参数能够使初始参数彼此之间的关系和与训练样本之间的关系相比于完全随机初始化更加合理。
  * 三种初始化方式下，训练过程验证集聚类精度曲线都有所不同：
      * 使用随机初始化的聚类精度曲线先升后降再升，这可能是因为随机初始化的参数位于一个很不稳定的局部最优点附近导致的。
      * 使用随机初始化后验的聚类精度曲线不断上升，但后面上升缓慢，100 次迭代后仍未收敛。
      * 使用 K-Means 初始化的聚类精度一开始就处于很高水平，之后变化缓慢，有先升后降的趋势，可能有一定程度的过拟合。

  * 随机初始化参数和随机初始化后验的模型训练耗时差不多，后者略多于前者，因为后者比前者多一个 M 步；使用 K-Means 初始化的模型训练耗时多于前两者，因为它需要额外训练一个 K-Means 模型。

* 综合以上分析，使用 K-Means 初始化的方法效果优于另外两者，它虽然耗时较长，但是只要将迭代次数减少为 60 左右，既能够减少耗时，也能够防止过拟合，得到更高的聚类精度。

#### 2.5.2 不同的协方差矩阵结构

|   协方差矩阵结构   |                           普通矩阵                           |                            对角阵                            |                       元素相等的对角阵                       |
| :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      训练耗时      |                           77.98 s                            |                           77.17 s                            |                           76.64 s                            |
| 最终测试集聚类精度 |                           60.07 %                            |                           49.16 %                            |                           53.86 %                            |
|   测试集聚类结果   | ![随机初始化参数](.\figures\GMM (random init, ordinary cov).jpg) | ![随机初始化后验](.\figures\GMM (random init, diag cov).jpg) | ![使用K-Means初始化](.\figures\GMM (random init, diag cov with the same element).jpg) |

* 训练过程验证集聚类精度曲线
    <img src=".\figures\random init.jpg" alt="训练过程验证集聚类精度曲线" style="zoom:50%;" />
* 测试集真实标签结果
    <img src=".\figures\gold_result.jpg" alt="测试集真实标签结果" style="zoom:50%;" />
* 结果分析
    * 使用普通矩阵要好于使用对角阵，使用元素相同的对角阵要好于使用元素不同的对角阵，从散点图也可看出使用 普通矩阵作为协方差矩阵的模型最终聚类结果与真实结果更加接近。这是因为限制协方差矩阵为对角阵意味着数据点各个维度之间相互独立，但实际上手写数字图像的像素点之间是有位置关联的，将协方差矩阵限制为对角阵限制了模型学习各个维度之间关联的能力，从而降低了模型的性能。
    * 三种结构的协方差矩阵训练过程聚类精度都有先将后升的趋势，这可能是因为参数初始化方式为随机初始化引起的，参数初始值可能位于一个很不稳定的局部最优点附近。
    * 这三种结构的协方差矩阵的模型训练耗时相近，因为不同结构的协方差矩阵并没有给训练过程的计算量带来很大差异。
* 综合以上分析，使用普通矩阵作为协方差矩阵的效果更好。

## 3 K-Means vs GMM

* 从模型性能上看，以上实验中 K-Means 模型聚类精度最高为 55.54%，GMM 模型最高为 64.1%，GMM 模型的聚类效果明显优于 K-Means 模型。这是因为 K-Means 只是简单地通过距离确定不同类的边界，没有考虑到类内数据点分布的特性；而 GMM 模型虽然是用简单的高斯分布来拟合各个类数据点的分布，但也比 K-Means 模型有更强的表达能力。实际上，K-Means 模型只是 GMM 模型协方差矩阵为单位阵时的一种特殊情况。
* 从训练时间上看，随机初始化的 K-Means 跟 GMM 训练耗时相近，而基于距离初始化的 K-Means 由于有较好的初始质心而耗时明显少于其他模型，而基于 K-Means 初始化的 GMM 模型因为要额外训练一个 K-Means 模型，耗时明显多于其他模型。
* 在实验中发现，K-Means 对原始 784 维的数据也能够进行聚类，聚类精度跟降到 50 维后的差不多；而 GMM 模型对于高维稀疏数据很敏感，很容易由于数值计算的误差导致协方差矩阵为奇异矩阵而使训练无法进行，使用 PCA 将数据降到 50 维后问题才得以解决。
* K-Means 只能对数据点所属类型进行判别，而使用 GMM 模型还能够获得数据点属于各个类别的后验概率，增强了模型的可解释性。

根据以上分析，总结两种模型的优缺点如下：

| 模型     | K-Means                                                      | GMM                                                          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **优点** | 算法简单，容易实现；<br />训练耗时短（使用基于距离的初始化）。 | 模型表现力强，聚类效果好；<br />能够给出数据点属于每个类别的概率，可解释性强。 |
| **缺点** | 模型表现力有限，无法对复杂分布的数据进行很好的聚类；<br />无法给出数据点属于每个类别的概率，可解释性差。 | 算法比较复杂；<br />对输入数据和计算过程的数值误差比较敏感。 |

