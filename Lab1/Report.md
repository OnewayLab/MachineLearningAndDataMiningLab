# 机器学习与数据挖掘作业 1

## 1 SVM 模型的一般理论

### 1.1 软线性最大间隔分类器

* 对线性不可分的训练数据使用线性分类器，将无法找出用于划分类别的超平面。为了解决这个问题，我们把约束条件放松为

    $$
    y^{(n)} \cdot\left(\boldsymbol{w}^{T} \boldsymbol{x}^{(n)}+b\right) \geq 1-\xi_{n}
    $$

    其中，$\xi_n\ge0$ 是松弛变量。

    目标函数变为

    $$
    \frac{1}{2}\|w\|^{2}+C \sum_{n=1}^{N} \xi_{n}
    $$

    其中，$C$ 用来控制相对重要性。

* 现在，优化问题变为

	$$
	\begin{aligned}
	&\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{n=1}^{N} \xi_{n} \\
	&\quad\text {s.t.: } y^{(n)} \cdot\left(\boldsymbol{w}^{T} \boldsymbol{x}^{(n)}+b\right) \geq 1-\xi_{n} \\
	&\quad\quad\quad\xi_{n} \geq 0, \quad \text { for } n=1,2, \cdots, N
	\end{aligned}
	$$

* 对偶问题为

	$$
	\begin{array}{l}
	\max _{\boldsymbol{a}} g(\boldsymbol{a})\\
	\text { s.t.: } a_{n} \geq 0, a_{n} \leq C\\
	\quad\quad\sum_{n=1}^{N} a_{n} y^{(n)}=0
	\end{array}
	$$

	其中，$g(\boldsymbol{a})=\sum_{n=1}^{N} a_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_{n} a_{m} y^{(n)} y^{(m)} \boldsymbol{x}^{(n) T} \boldsymbol{x}^{(m)}$

* 得到最优的 $\boldsymbol{x}^*$ 和 $b^*$ 后，样本 $\boldsymbol{x}$ 可以这样分类

	$$
	\hat{y}(\boldsymbol{x})=\operatorname{sign}\left(\boldsymbol{w}^{* T} \boldsymbol{x}+b^{*}\right)
	$$

* 得到最优的 $\boldsymbol{a}^*$ 后，样本 $\boldsymbol{x}$ 可以这样分类

	$$
	\hat{y}(x)=\operatorname{sign}\left(\sum_{n=1}^{N} a_{n}^{*} y^{(n)} x^{(n) T} x+b^{*}\right)
	$$

	同样地，$\boldsymbol{a}^*$ 是稀疏的，只有位于间隔内的元素非 0。


###  1.2 非线性化

* 以上讨论的最大间隔分类器都是线性的。

* 为了使模型非线性化，我们通过一个基函数 $\phi$ 把原始数据 $x$ 映射到特征空间，那么最初的最大间隔优化问题变成

	$$
	\begin{aligned}
	&\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{n=1}^{N} \xi_{n} \\
	&\quad\text {s.t.: } y^{(n)} \cdot\left(\boldsymbol{w}^{T} \boldsymbol{\phi}(\boldsymbol{x}^{(n)})+b\right) \geq 1-\xi_{n} \\
	&\quad\quad\quad\xi_{n} \geq 0, \quad \text { for } n=1,2, \cdots, N
	\end{aligned}
	$$

	分类器：$\hat{y}(\boldsymbol x)=s i g n{\bigl(}\boldsymbol w^{*T}\boldsymbol\phi{\bigl(}\boldsymbol x^{(n)}{\bigr)}+b^{*}{\bigr)}$

* 直觉上，数据在高维空间中更容易分类，所以为了得到更好的分类效果，我们希望特征空间的维度越高越好。然而，如果基函数 $\phi(\boldsymbol x)$ 的维度过高，主优化问题会非常难解，但是我们可以通过它的对偶问题来求解：

	$$
	\begin{aligned}
	\max _{\boldsymbol{a}} g(\boldsymbol{a}) & \\
	\text { s.t.: } & a_{n} \geq 0, a_{n} \leq C \\
	& \sum_{n=1}^{N} a_{n} y^{(n)}=0
	\end{aligned}
	$$

	其中，$g(\boldsymbol{a})=\sum_{n=1}^{N} a_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_{n} a_{m} y^{(n)} y^{(m)} \phi\left(x^{(n)}\right)^{T} \phi\left(x^{(m)}\right)$

	分类器：
	$$
	\hat y(\boldsymbol x)=sign\left (\sum_{n=1}^Na_n^*y^{(n)}\boldsymbol \phi(\boldsymbol x^{(n)})^T\boldsymbol\phi(\boldsymbol x)+b^*\right )
	$$
	但是对偶公式需要计算 $\phi(x^{(n)})^{T}\phi(x)$，这在高维空间很难计算。这个问题可以使用 kernel trick 解决。

### 1.3 核函数

* 核函数：能够表达为某个函数的内积的双变量函数：

	$$
	k(\boldsymbol x, \boldsymbol x')=\boldsymbol\phi(\boldsymbol x)^T\boldsymbol\phi(\boldsymbol x')
	$$

* Mercer 定理：

	如果一个函数 $k(\boldsymbol x, \boldsymbol x')$ 是对称正定的，即

	$$
	\iint g(\boldsymbol{x}) k(\boldsymbol{x}, \boldsymbol{y}) g(\boldsymbol{y}) d \boldsymbol{x} d \boldsymbol{y} \geq 0, \quad \forall g(\cdot) \in L^{2}
	$$

	那么一定存在一个函数 $\boldsymbol{\phi}(\cdot)$ 使得 $k(\boldsymbol x, \boldsymbol x')=\boldsymbol\phi(\boldsymbol x)^T\boldsymbol\phi(\boldsymbol x')$。

	即：如果一个函数 $k(\boldsymbol x, \boldsymbol x')$ 满足对称正定条件，那么它一定是一个核函数。

* 一个最常使用的核函数是**高斯核函数**：

	$$
	k\left(x, x^{\prime}\right)=\exp \left\{-\frac{1}{2 \sigma^{2}}\left\|x-x^{\prime}\right\|^{2}\right\}
	$$

	高斯核的 $\boldsymbol{\phi(\cdot)}$ 有无穷维：

	$$
	\phi(x)=e^{-x^{2} / 2 \sigma^{2}}\left[1, \sqrt{\frac{1}{1 ! \sigma^{2}}} x, \sqrt{\frac{1}{2 ! \sigma^{4}}} x^{2}, \sqrt{\frac{1}{3 ! \sigma^{6}}} x^{3}, \cdots\right]^{T}
	$$

### 1.4 Kernel Trick

* Kernel trick：使用核函数 $k(\boldsymbol x, \boldsymbol x')$ 代替 $\boldsymbol\phi(\boldsymbol x)^T\boldsymbol\phi(\boldsymbol x')$

* 有了核函数，对偶最大化间隔分类器的优化问题可以写成

	$$
	\begin{aligned}
	\max _{\boldsymbol{a}} g(\boldsymbol{a}) & \\
	\text { s.t.: } & a_{n} \geq 0, a_{n} \leq C \\
	& \sum_{n=1}^{N} a_{n} y^{(n)}=0
	\end{aligned}
	$$

	其中，$g(\boldsymbol{a})={\sum_{n=1}^{N} a_{n}-\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_{n} a_{m} y^{(n)} y^{(m)} k\left (\boldsymbol x^{(n)}, \boldsymbol{x} ^{(m)} \right )}$

* 分类器：

	$$
	\hat{y}(\boldsymbol{x})=\operatorname{sign}\left(\sum_{n=1}^{N} a_{n}^{*} y^{(n)} k\left(x^{(n)}, x\right)+b^{*}\right)
	$$

## 2 采用不同核函数的模型和性能比较及分析



## 3 采⽤hinge loss的线性分类模型和SVM模型之间的关系

## 4 采⽤hinge loss线性分类模型和cross-entropy loss线性分类模型⽐较

## 5 训练过程（包括初始化⽅法、超参数参数选择、⽤到的训练技巧等）

## 6 实验结果、分析及讨论
