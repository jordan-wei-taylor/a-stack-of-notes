[Descent Methods](../descent-methods.md)

## Steepest Descent

#steepest-descent #descent-methods

The method of steepest descent considers only the current gradient when choosing the search direction $\mathbf{p}_k$. This method can be thought of as being *memoryless* as it does not utilise the history of search directions taken i.e. $\mathbf{p}_1,...,\mathbf{p}_{k-1}$. By differentiating the loss function stated at the start of the section $\mathcal{L}(\mathbf{x}_{k}):=||\mathbf{A}\mathbf{x}_{k} - \mathbf{b}||_2^2$, we know the gradient is proportional to $\mathbf{A}^\text{T}\mathbf{r}_{k}$ where $\mathbf{r}_k := \mathbf{b} - \mathbf{A}\mathbf{x}_k$. At this point, we digress to the case where $\mathbf{A}\in\mathbb{R}^{m,m},\ \mathbf{x}\in\mathbb{R}^m$ and $\mathbf{b}\in\mathbb{R}^m$ such that we have a square system of linear equations. The theory that we state can be then applied on the original non-square problem by multiplying $\mathbf{A}^\text{T}$ from the left i.e. $\mathbf{A}^\text{T}\mathbf{A}\mathbf{x} = \mathbf{A}^\text{T}\mathbf{b}$.


### Steepest Descent Algorithm

$$
\begin{flalign}
	&\text{Require:}\\
	&\qquad\text{data } \mathbf{A} \text{ and } \mathbf{b}\\
	&\qquad\text{sufficiently small learning rate } \alpha > 0\\
	&\qquad\text{sufficiently small acceptence threshold } \tau > 0\\\\
	&\text{Initialise:}\\
	&\qquad\mathbf{x}_0 \text{ arbitrarily}\\\\
	&\text{For } k = \{1,\ 2,\ ...\} \text{ do:}\\\\
	&(1) &\mathbf{r}_{k - 1} &= \mathbf{Ax}_{k - 1} - \mathbf{b}&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\\
	&(2) &\mathbf{p}_k &= \mathbf{r}_{k - 1}\\
	&(3) &\mathbf{x}_k &= \mathbf{x}_{k - 1} + \alpha \mathbf{p}_k\\
	&(4) &\text{terminate } &\text{if } ||\mathbf{x}_k - \mathbf{x}_{k - 1}||_2 = \alpha||\mathbf{p}_k||_2 \le \tau
\end{flalign}
$$

^steepest-descent-algorithm

A common questions is how to determine the optimal learning rate?  From [Determining the Optimal Learning Rate](steepest-descent/determining-the-optimal-learning-rate.md), we can choose the optimal learning rate at every iteration of the [Steepest Descent Algorithm](#^steepest-descent-algorithm). 


### Steepest Descent with Optimal Learning Rate Algorithm

$$
\begin{flalign}
	&\text{Require:}\\
	&\qquad\text{data } \mathbf{A} \text{ and } \mathbf{b}\\
	&\qquad\text{sufficiently small acceptence threshold } \tau > 0\\\\
	&\text{Initialise:}\\
	&\qquad\mathbf{x}_0 \text{ arbitrarily}\\\\
	&\text{For } k = \{1,\ 2,\ ...\} \text{ do:}\\\\
	&(1) &\mathbf{r}_{k - 1} &= \mathbf{Ax}_{k - 1} - \mathbf{b}&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\\
	&(2) &\mathbf{p}_k &= \mathbf{r}_{k - 1}\\
	&(3) &\alpha_k &= ||\mathbf{p}_{k}||_2^2\ /\ ||\mathbf{p}_{k}||_\mathbf{A}^2\\
	&(4) &\mathbf{x}_k &= \mathbf{x}_{k - 1} + \alpha_k \mathbf{p}_k\\
	&(5) &\text{terminate } &\text{if } ||\mathbf{x}_k - \mathbf{x}_{k - 1}||_2 = \alpha_k||\mathbf{p}_{k}||_2 \le \tau
\end{flalign}
$$

^steepest-descent-algorithm-optimal


