[A Stack of Notes](../../../../a-stack-of-notes.md) / [Miscellaneous](../../../miscellaneous.md)  / [Descent Methods](../../descent-methods.md) / [Steepest Descent](../steepest-descent.md)

### Determining the Optimal Learning Rate

Here we determine the optimal learning rate $\alpha_k$ by considering the following functional:

$$
\begin{equation}
	\phi(\mathbf{x}) := \frac{1}{2}\mathbf{x}^\text{T}\mathbf{A}\mathbf{x} - \mathbf{x}^\text{T}\mathbf{b}.\tag{2}
\end{equation}
$$
We see that by considering the derivative, the minimisation of $\phi$ occurs when $\mathbf{A}\mathbf{x} = \mathbf{b}$. By considering some arbitrary $\mathbf{y}_k$ we can write $\phi(\mathbf{x})$ in terms of $\mathbf{y}_k$ and a couple of other terms:

$$
\begin{align}
\phi(\mathbf{x}_k) &= \phi(\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k)\nonumber\\

&= \frac{1}{2}(\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k)^\text{T}\mathbf{A}(\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k) - (\mathbf{y}_k + \mathbf{x}_k - \mathbf{y}_k)^\text{T}\mathbf{b}\\

&= \phi(\mathbf{y}_k) + \frac{1}{2}||\mathbf{x}_k - \mathbf{y}_k||_\mathbf{A}^2 - (\mathbf{x}_k - \mathbf{y}_k)^\text{T}\mathbf{r}(\mathbf{y}_k).\quad(\text{let } \mathbf{y}_k = \mathbf{x}_{k - 1} \text{ and recall that }\mathbf{x}_k = \mathbf{x}_{k - 1} + \alpha_k\mathbf{p}_k)\\\\



\phi(\mathbf{x}_k) &= \phi(\mathbf{x}_{k-1}) + \frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\\

&= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\alpha_k^2 - 2\alpha_k\frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)\\

&= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\left(\alpha_k - \frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)^2 - \frac{(\mathbf{p}_k^\text{T}\mathbf{r}_{k-1})^2}{||\mathbf{p}_k||_\mathbf{A}^4}\right)\\

&= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\alpha_k - \frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)^2 - \frac{(\mathbf{p}_k^\text{T}\mathbf{r}_{k-1})^2}{2||\mathbf{p}_k||_\mathbf{A}^2}

\end{align}
$$
What this result shows is that regardless of the search direction $\mathbf{p}_k$, we can choose $\alpha_k$ to ensure the cost function $\phi$ is being minimised by setting $\alpha_k = \mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\ /\ ||\mathbf{p}_k||_\mathbf{A}^2$. In the case of steepest descent, we set $\mathbf{p}_k = \mathbf{r}_{k-1}$. An interesting application of the above provides an [Upper Bound on the Convergence Rate](upper-bound-convergence-rate.md).

