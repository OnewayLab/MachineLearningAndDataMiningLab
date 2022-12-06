# 机器学习与数据挖掘作业 3

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

### 1.3 实验结果与分析

## 2 GMM

### 2.1 算法原理

### 2.2 初始化方法

### 2.3 实验结果与分析

## 3 K-Means vs GMM
