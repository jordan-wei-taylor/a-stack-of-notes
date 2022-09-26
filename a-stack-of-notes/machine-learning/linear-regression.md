[A Stack of Notes](a-stack-of-notes) / [Machine Learning](a-stack-of-notes/machine-learning.md) / [Linear Regression](linear-regression)


# Linear Regression

#linear #regression #linear-regression #ordinary-least-squares

Assume we have a supervised dataset $D = \{(\mathbf{x}_{1},\ y_1),\ ...,\ (\mathbf{x}_n,\ y_n)\}$ where $\mathbf{x}_i\in\mathbb{R}^m$ and $y_i\in\mathbb{R}$ with $n$ being the total number of observation - target pairs in the dataset, and $m$ the number of features. Then, the Linear Regression model is defined as
$$
\begin{align}
	y_i &= w_0 + \mathbf{x}_i^\text{T}\mathbf{w} + \epsilon_i, \quad \epsilon_i\overset{iid}{\sim}\mathcal{N}(0, \sigma^2),\tag{1}
\end{align}
$$
where $w_0$ is the intercept, $\mathbf{w}\in\mathbb{R}^m$ is a vector of linear coefficients, and  $\epsilon_i$ is a random error term.

![](demo.png)

[Bayesian Linear Regression](bayesian-linear-regression)
[Sparse Bayesian Linear Regression](sparse-bayesian-linear-regression)
