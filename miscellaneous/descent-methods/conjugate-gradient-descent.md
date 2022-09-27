[Descent Methods](../descent-methods.md)

### Conjugate Gradient Descent
#conjugate-gradient-descent

The method of conjugate gradient utilises past search directions when selecting the next search direction. At the $k$-th iteration not only do we know the current gradient $\mathbf{r}_{k-1}$, we know all the previous gradients $\{\mathbf{r}_0,...,\mathbf{r}_{k-2}\}$. By utilising this information we can search in some orthogonal space and converge much quicker than the method of steepest descent.

The general idea behind this algorithm is: since $\alpha_k := \mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\ /\ ||\mathbf{p}_k||_\mathbf{A}^2$, we are looking for a search direction $\mathbf{p}_k$ where $\mathbf{p}_k \neq \mathbf{r}_{k-1}$, as in the case of steepest descent, and $\mathbf{p}_k^\text{T}\mathbf{r}_{k-1} \neq 0$. We want $\mathbf{p}_k$ and $\mathbf{x}_k$ to satisfy the following conditions:


+ $\mathbf{p}_1,...,\mathbf{p}_k$ should be linearly independent.

+ $\phi(\mathbf{x}_k) = \min_{\mathbf{x}\in\mathbf{x}_0 + \text{span}\{\mathbf{p}_1,...,\mathbf{p}_k\}}\phi(\mathbf{x})$.

+  $\mathbf{x}_k$ can be calculated easily from $\mathbf{x}_{k-1}$.


Consider the iterative update equation for $\mathbf{x}_k$:

$$
\begin{align}

\mathbf{x}_1 &= \mathbf{x}_0 + \alpha_1\mathbf{p}_1\\

\mathbf{x}_2 &= \mathbf{x}_1 + \alpha_2\mathbf{p}_2 = \mathbf{x}_0 + \alpha_1\mathbf{p}_1 + \alpha_2\mathbf{p}_2\\

\vdots\ \  &\\

\mathbf{x}_k &= \mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k

\end{align}
$$

where $\mathbf{P}_{k-1} = [\mathbf{p}_1,...,\mathbf{p}_{k-1}]$ with parameters $\mathbf{y}_k$ and $\alpha_k$. The objective is to determine the parameters $\mathbf{y}_k$ and $\alpha_k$:

$$
\begin{align}

\phi(\mathbf{x}_k) &= \frac{1}{2}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k)^\text{T}\mathbf{A}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k) - (\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k + \alpha_k\mathbf{p}_k)^\text{T}\mathbf{b}\\

&= \phi(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k) + \alpha_k\mathbf{p}_k^\text{T}\mathbf{A}(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k) - \alpha_k\mathbf{p}_k^\text{T}\mathbf{b} + \frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2\\

&= \textcolor{blue}{\phi(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k)} + \alpha_k\mathbf{p}_k^\text{T}\mathbf{A}\mathbf{P}_{k-1}\mathbf{y}_k + \textcolor{red}{\frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_0},\quad \text{as } \mathbf{b} - \mathbf{A}\mathbf{x}_0 = \mathbf{r}_0

\end{align}
$$

We tried to separate the $\textcolor{blue}{\mathbf{y}_k}$ and $\textcolor{red}{\alpha}$ terms in our calculations but have a mixed middle term. If we did not have this mixed term in the middle, we could just minimise over the two variables separately. Hence, we choose $\mathbf{p}_k$ such that:

$$
\begin{equation}

\mathbf{p}_k^\text{T}\mathbf{A}\mathbf{P}_{k-1} = \mathbf{0}

\end{equation}
$$

and we are left with the following minimisation task:

$$
\begin{equation}

\min_{\mathbf{x}_k\in\mathbf{x}_0+\text{span}\{\mathbf{p}_1,...,\mathbf{p}_k\}}\phi(\mathbf{x}_k) = \min_{\mathbf{y}_k}\big(\phi\left(\mathbf{x}_0 + \mathbf{P}_{k-1}\mathbf{y}_k\right)\big) + \min_{\alpha_k}\left(\frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_0\right)

\end{equation}
$$

By completing the square, we can optimally compute $\alpha_k$ for any search direction $\mathbf{p}_k$. The search direction $\mathbf{p}_k$ is defined by the Gram-Schmidt process:
$$
\begin{equation}
	\mathbf{p}_k = \mathbf{r}_{k - 1} - \sum_{j = 1}^{k - 1} \frac{\langle\mathbf{r}_{k - 1},\ \mathbf{p}_j\rangle_\mathbf{A}}{||\mathbf{p}_j||_\mathbf{A}^2}
\end{equation}
$$
The above ensures that the $k$-th search direction is orthogonal to all the previous search directions.


### The Conjugate Gradient Method

$$
\begin{flalign}
	&\text{Require:}&&&&&&&&&&&&&\\
	&\qquad\text{Data } \mathbf{A},\ \mathbf{b}&\\\\
	&\text{Initialise:}&\\
	&\qquad\mathbf{x}_0 \text{ arbitrarily}&\\
	&\qquad\mathbf{p}_0 = \mathbf{0}&\\\\
	&\text{For } k = \{1,\ 2,\ ...\} \text{ do:}&\\
	&(1) &\beta_{k - 1} &= \langle\mathbf{r}_{k - 1},\ \mathbf{p}_{k - 1}\rangle_\mathbf{A}\ /\ ||\mathbf{p}_{k - 1}||_\mathbf{A}^2\\
	&(2) &\mathbf{p}_k &= \mathbf{r}_{k - 1} - \beta_{k - 1}\mathbf{p}_{k - 1}\\
	&(3) &\alpha_k &= ||\mathbf{r}_{k - 1}||_2^2\ /\ ||\mathbf{p}_k||_\mathbf{A}^2\\
	&(4) &\mathbf{x}_k &= \mathbf{x}_k + \alpha_k\mathbf{p}_k\\
	&(5) &\text{check } &\text{for convergence}
 \end{flalign}
$$