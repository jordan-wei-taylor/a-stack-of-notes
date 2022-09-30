[A Stack of Notes](../../a-stack-of-notes.md) / [Machine Learning](../machine-learning.md) / [Linear Regression](linear-regression.md)


# Linear Regression

#linear #regression #linear-regression #ordinary-least-squares

Assume we have a supervised dataset $D = \{(\mathbf{x}_{1},\ y_1),\ ...,\ (\mathbf{x}_n,\ y_n)\}$ where $\mathbf{x}_i\in\mathbb{R}^m$ and $y_i\in\mathbb{R}$ with $n$ being the total number of observation - target pairs in the dataset, and $m$ the number of features. Then, the Linear Regression model assumes
$$
\begin{align}
	y_i &= w_0 + \mathbf{x}_i^\text{T}\mathbf{w} + \epsilon_i, \quad \epsilon_i\overset{iid}{\sim}\mathcal{N}(0, \sigma^2),\tag{1}
\end{align}
$$

^linear-formulation

where $w_0$ is the intercept, $\mathbf{w}\in\mathbb{R}^m$ is a vector of linear coefficients, and  $\epsilon_i$ is an independent random error term which incorperates the noise present in the data. For notational convenience it is common to write $w_0 + \mathbf{x}_i^\text{T}\mathbf{w}$ as $\mathbf{x}_i^\text{T}\mathbf{w}$ by redefining $\mathbf{x}_i \leftarrow [1, \mathbf{x}_i]^\text{T}$ and $\mathbf{w} \leftarrow [w_0, \mathbf{w}]^\text{T}$ such that $\mathbf{y}_i = \mathbf{x}_i^\text{T}\mathbf{w} + \epsilon_i$. We use [Eq. (1)](#^linear-formulation) to formulate the likelihood function

$$
\begin{align}
	y_i &\overset{iid}{\sim}\mathcal{N}(\mathbf{x}_i^\text{T}\mathbf{w},\ \sigma^2)\\\\
	 p(\mathbf{y}\mid \mathbf{X},\ \mathbf{w},\ \sigma^2\mathbf{I}) &= \mathcal{N}(\mathbf{y}\mid \mathbf{Xw},\ \sigma^2\mathbf{I})\\\\
	 &= (2\pi)^{-n/2} |\sigma^2\mathbf{I}|^{-1/2}\exp\big(-(\mathbf{y} - \mathbf{Xw})^\text{T}(\sigma^2\mathbf{I})^{-1}(\mathbf{y} - \mathbf{Xw}) / 2\big)\\\\
	 \log p(\mathbf{y}\mid \mathbf{X},\ \mathbf{w},\ \sigma^2\mathbf{I}) &= -\frac{n}{2}\log 2\pi \sigma^2 - \frac{1}{2\sigma^2}||\mathbf{y} - \mathbf{Xw}||_2^2
\end{align}
$$

If the noise is unknown, we can estimate $\mathbf{w}$ by maximising the above yielding 

$$
\begin{align}
	\mathbf{\hat{w}} = \underset{\mathbf{w}}{\arg \min}\ ||\mathbf{y} - \mathbf{Xw}||_2^2.
\end{align}
$$

Using some calculus, it can be shown that our estimate can be computed by solving the following system of linear equations for $\mathbf{w}$

$$
\begin{align}
	\mathbf{X}^\text{T}\mathbf{Xw} = \mathbf{X}^\text{T}\mathbf{y}
\end{align}
$$

We can compute the estimate of $\mathbf{w}$ so long as $\mathbf{X}^\text{T}\mathbf{X}$ is non-singular. Further, if  $\mathbf{X}^\text{T}\mathbf{X}$ is ill-posed, computing the inverse may be not accurate so regularisation is required to ensure $||\mathbf{w}||$ does not explode.

In the event that the feature space is large, we may wish to avoid computing the inverse as it is computationally expensive. Instead we would resort to a [Descent Method](../../miscellaneous/descent-methods.md).

See also:
+ [Bayesian Linear Regression](linear-regression/bayesian-linear-regression.md)
+ [Sparse Bayesian Linear Regression](linear-regression/sparse-bayesian-linear-regression.md)
