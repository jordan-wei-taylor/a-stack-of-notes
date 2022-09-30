[A Stack of Notes](../../../../a-stack-of-notes.md) / [Miscellaneous](../../../miscellaneous.md)  / [Descent Methods](../../descent-methods.md) / [Steepest Descent](../steepest-descent.md)


# Upper Convergence Bound

Define the error vector $\mathbf{e}_k:=\mathbf{x}_*-\mathbf{x}_k$ and consider the following:

$$
\begin{align}

\mathbf{x}_{k +1} &= \mathbf{x}_{k} + \alpha\mathbf{r}_{k}\\

&= \mathbf{x}_k +\alpha(\mathbf{b} - \mathbf{A}\mathbf{x}_k)\\

&= \mathbf{x}_k +\alpha\mathbf{A}(\mathbf{x}_* - \mathbf{x}_k\\

\Rightarrow \mathbf{x}_* - \mathbf{x}_{k+1} &= (1 - \alpha\mathbf{A})(\mathbf{x}_* - \mathbf{x}_k)\\

\mathbf{e}_{k+1} &= (1 - \alpha\mathbf{A})\mathbf{e}_k\\

\end{align}
$$
By considering $||\mathbf{e}_{k+1}||_\mathbf{A}^2$ and expanding $\mathbf{e}_k=\sum_{j=1}^n a_j\mathbf{z}_j$ w.r.t. orthogonal basis of eigenvectors of $\mathbf{A}$ for some coefficients $\{a_j\}_{j=1}^n \subset \mathbb{R}$ we obtain

$$
\begin{align}
\Rightarrow ||\mathbf{e}_{k + 1}||_\mathbf{A}^2 &= \mathbf{e}_k^\text{T}(1 - \alpha \mathbf{A})^\text{T}\mathbf{A}(1 - \alpha \mathbf{A})\mathbf{e}_k\\


||\mathbf{e}_{k+1}||_\mathbf{A}^2 &= \sum_{j=1}^m \lambda_ja_j^2(1 - \alpha\lambda_j)^2,\qquad \text{conveniently set $\alpha = 2 / (\lambda_1 + \lambda_m)$}\\

&= \sum_{j=1}^m \lambda_ja_j^2\left(\frac{\lambda_1 + \lambda_m - 2\lambda_j}{\lambda_1 + \lambda_m}\right)^2\\

&\leq \left(\frac{\lambda_1 - \lambda_m}{\lambda_1 + \lambda_m}\right)^2\sum_{j=1}^m\lambda_j a_j^2\\

&= \left(\frac{\lambda_1 - \lambda_m}{\lambda_1 + \lambda_m}\right)^2 ||\mathbf{e}_{k}||_\mathbf{A}^2\\

\Rightarrow ||\mathbf{e}_k||_\mathbf{A} &\le \left(\frac{\lambda_1 - \lambda_m}{\lambda_1 + \lambda_m}\right)^k
||\mathbf{e}_0||_\mathbf{A}

\end{align}
$$

We see that the $\mathbf{A}$-norm of $\mathbf{e}_k$ shows convergence with an upper bound based on the eigenvalues of $\mathbf{A}$. We conveniently set $\alpha$ to help deduce the convergence bound so we know that by selecting the optimal $\alpha$, we have at least better convergence as stated. For well conditioned $\mathbf{A}$ i.e. $\lambda_1$ is close to $\lambda_m$, the convergence is fast but for ill-conditioned $\mathbf{A}$ i.e. $\lambda_1 \gg \lambda_m$ in the presence of highly correlated features, the convergence rate is much slower.