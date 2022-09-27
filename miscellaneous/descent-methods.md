[A Stack of Notes](../a-stack-of-notes.md)

# Descent Methods

#descent-methods

Consider the normal equation which seeks a quantity $\mathbf{x}$ such that we minimise the common measure of loss:
$$
\begin{equation}
\mathcal{L}(\mathbf{x}) := ||\mathbf{Ax} - \mathbf{b}||_2^2 = ||\mathbf{r}||_2^2,\tag{1}
\end{equation}
$$

^l2-loss

where $\mathbf{r}$ is known as the residual error vector and is defined as $\mathbf{r}(\mathbf{x}) := \mathbf{Ax} - \mathbf{b}$. We constrain ourselves against computing this quantity using the analytic solution as described in [Orinary Least-Squares Regression](../a-stack-of-notes/machine-learning/linear-regression.md) and instead, seek an iterative method that updates our $k$-th guess of the true solution. These class of methods are commonly known as *descent* methods as they all consider the gradient of [Eq. (1)](#^l2-loss) when minimising it. For generality, they all have the following update rule:
$$\begin{equation}
	\mathbf{x}_k = \mathbf{x}_{k-1} + \alpha_k\mathbf{p}_k, \tag{2}
\end{equation}$$
where $\alpha_k$ is known as the *learning rate* or *step size* and $\mathbf{p}_k$ as the search direction. Below we state the few desirable properties we would like $\mathbf{x}_k$ to have:

+ **Existence.** What this means is there exists some quantity $\mathbf{x}_k$ such that $||\mathbf{x}_k||_2^2 < \infty$ i.e. we do not want a solution that explodes. Why this is a good property to have becomes clearer when we consider equations in a physical context where $\mathbf{x}_k$ may be bounded between reasonably sized numbers.

+ **Uniqueness.** Given $\mathbf{x}_0$ is an arbitrary initialisation, for any $\mathbf{x}_0$, $\mathbf{x}_k \rightarrow \mathbf{x}_*$ as $k \rightarrow \infty$ for some $\mathbf{x}_*$. It should be noted that $\mathbf{x}_*$ may not be $\mathbf{x}_{\text{ols}}$ as described in the analytic solution but is important that $\mathbf{x}_*$ is unique to give clear interpretability.

+ **Stability.** Given an observation matrix $\mathbf{A}$ and some response vector $\mathbf{b}$, if we perturb $\mathbf{A}$ and / or $\mathbf{b}$ by some small noise i.e. use the data $\mathbf{\tilde{A}}$ and $\mathbf{\tilde{b}}$, where the tilde notation represents a noisy representation of their respective non-noisy counter-parts, the solution $\mathbf{\tilde{x}}_*$ should satisfy $||\mathbf{\tilde{x}}_* - \mathbf{x}_*||_2^2 \le \epsilon$ for some reasonably small $\epsilon$.

+ **Monotonicity.** We desire that with every iteration, we are closer to the true underlying solution i.e. $||\mathbf{x}_k - \mathbf{x}_*||_2^2 < ||\mathbf{x}_{k-1} - \mathbf{x}_*||_2^2\ \forall\ k\in\{1, 2, 3,...\}$.

We now consider the most basic descent method which is then used as a foundation for many more sophisticated descent methods.

1.  [Steepest Descent](descent-methods/steepest-descent.md)
2.  [Conjugate Gradient Descent](descent-methods/conjugate-gradient-descent.md)

