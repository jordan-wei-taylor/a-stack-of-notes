[A Stack of Notes](a-stack-of-notes.md) / [Descent Methods](descent-methods.md)

# Descent Comparison Example

#descent-methods #steepest-descent  #conjugate-gradient-descent 

Often with real-world datasets, the number of features is large and so computing the analytic solution is expensive due to the inversion term as shown below
$$
\begin{align}
	\hat{\mathbf{w}} = \big(\mathbf{X}^\text{T}\mathbf{X}\big)^{-1}\mathbf{X}^\text{T}\mathbf{y}. \tag{1}
\end{align}
$$
Additionally, the data tends to be correlated resulting in steepest descent requiring many iterations to converge.  For visualisation purposes, suppose that $\mathbf{x}_i \overset{iid}{\sim} \mathcal{N}\bigg(\begin{bmatrix}0\\0\end{bmatrix}, \begin{bmatrix}2 & 1\\1 & 2\end{bmatrix}\bigg)$ and that $\mathbf{y}_i = \mathbf{x}_i^\text{T}\mathbf{w}$ with $\mathbf{w} = [5,\ -8]^\text{T}$, i.e. our observations are correlated and our target variable is a deterministic (noise-less) linear function of our observed feature values. 

<div style="margin-left: auto !important; margin-right: auto !important; width: 70%"> <div src="../../a-stack-of-notes/_assets/descent-methods/toy-data.png" class="internal-embed"></div> </div>
<p style="font-size:15;text-align:center">Figure 1. <i>Sampled data points on a 2d plane.</i></p>

^d17ba5

Given the data relatively sparse (but sufficient) number of data points above, we compare how the method of [Steepest Descent](steepest-descent.md), both the naive and optimal setting of the learning rate, against [Conjugate Gradient Descent](conjugate-gradient-descent.md).

<div style="margin-left: auto !important; margin-right: auto !important; width: 70%"> <div src="../../a-stack-of-notes/_assets/descent-methods/comparison.png" class="internal-embed"></div> </div><p style="font-size:15;text-align:center">Figure 2. <i>Steepest descent and conjugate gradient descent solution path starting from the origin.</i></p>


From the above, we see that the [Conjugate Gradient Descent](conjugate-gradient-descent.md) method only takes two iterations to converge. In higher dimensions, this is most preferred to the [Steepest Descent](steepest-descent.md) method.