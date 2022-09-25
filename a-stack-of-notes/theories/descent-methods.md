[Theories](../theories.md)

# Descent Methods

#descent-methods

Consider the normal equation which seeks a quantity $\mathbf{x}$ such that we minimise the common measure of loss:

$$\begin{equation}
\mathcal{L}(\mathbf{x}) := ||\mathbf{Ax} - \mathbf{b}||_2^2 = ||\mathbf{r}||_2^2,\tag{D1}
\end{equation}
$$where $\mathbf{r}$ is known as the residual error vector and is defined as $\mathbf{r}(\mathbf{x}) := \mathbf{Ax} - \mathbf{b}$. We constrain ourselves against computing this quantity using the analytic solution and instead seek an iterative method that updates our $k$-th guess of the true solution. This class of methods are commonly known as *descent* ,etjpds as they all consider the gradient of D1 when minimising it. For generality, they all have the following update rule:
$$\begin{equation}
	\mathbf{x}_k = \mathbf{x}_{k-1} + \alpha_k\mathbf{p}_k, \tag{1}
\end{equation}$$
where $\alpha_k$ is known as the \textit{learning rate} or \textit{step size} and $\mathbf{p}_k$ as the search direction. Below we state the few desirable properties we would like $\mathbf{x}_k$ to have:

+ **Existence.** What this means is there exists some quantity $\mathbf{x}_k$ such that $||\mathbf{x}_k||_2^2 < \infty$ i.e. we do not want a solution that explodes. Why this is a good property to have becomes clearer when we consider equations in a physical context where $\mathbf{x}_k$ may be bounded between reasonably sized numbers.

+ **Uniqueness.** Given $\mathbf{x}_0$ is an arbitrary initialisation, for any $\mathbf{x}_0$, $\mathbf{x}_k \rightarrow \mathbf{x}_*$ as $k \rightarrow \infty$ for some $\mathbf{x}_*$. It should be noted that $\mathbf{x}_*$ may not be $\mathbf{x}_{\text{ols}}$ as described in the analytic solution but is important that $\mathbf{x}_*$ is unique to give clear interpretability.

+ **Stability.** Given an observation matrix $\mathbf{A}$ and some response vector $\mathbf{b}$, if we perturb $\mathbf{A}$ and / or $\mathbf{b}$ by some small noise i.e. use the data $\mathbf{\tilde{A}}$ and $\mathbf{\tilde{b}}$, where the tilde notation represents a noisy representation of their respective non-noisy counter-parts, the solution $\mathbf{\tilde{x}}_*$ should satisfy $||\mathbf{\tilde{x}}_* - \mathbf{x}_*||_2^2 \le \epsilon$ for some reasonably small $\epsilon$.

+ **Monotonicity.** We desire that with every iteration, we are closer to the true underlying solution i.e. $||\mathbf{x}_k - \mathbf{x}_*||_2^2 < ||\mathbf{x}_{k-1} - \mathbf{x}_*||_2^2\ \forall\ k\in\{1, 2, 3,...\}$.

We now consider the most basic descent method which is then used as a foundation for many more sophisticated descent methods.



## Steepest Descent

#steepest-descent 

The method of steepest descent considers only the current gradient when choosing the search direction $\mathbf{p}_k$. This method can be thought of as being *memoryless* as it does not utilise the history of search directions taken i.e. $\mathbf{p}_1,...,\mathbf{p}_{k-1}$. By differentiating the loss function stated at the start of the section $\mathcal{L}(\mathbf{x}_{k}):=||\mathbf{A}\mathbf{x}_{k} - \mathbf{b}||_2^2$, we know the gradient is proportional to $\mathbf{A}^\text{T}\mathbf{r}_{k}$ where $\mathbf{r}_k := \mathbf{b} - \mathbf{A}\mathbf{x}_k$. At this point, we digress to the case where $\mathbf{A}\in\mathbb{R}^{m,m},\ \mathbf{x}\in\mathbb{R}^m$ and $\mathbf{b}\in\mathbb{R}^m$ such that we have a square system of linear equations. The theory that we state can be then applied on the original non-square problem by multiplying $\mathbf{A}^\text{T}$ from the left i.e. $\mathbf{A}^\text{T}\mathbf{A}\mathbf{x} = \mathbf{A}^\text{T}\mathbf{b}$.

### Determining the optimal $\alpha$

Here we determine the optimal learning rate $\alpha_k$ by considering the following functional:

$$
\begin{equation}
	\phi(\mathbf{x}) := \frac{1}{2}\mathbf{x}^\text{T}\mathbf{A}\mathbf{x} - \mathbf{x}^\text{T}\mathbf{b}.\tag{2}
\end{equation}
$$
We see that by considering the derivative, the minimisation of $\phi$ occurs when $\mathbf{A}\mathbf{x} = \mathbf{b}$. By considering some arbitrary $\mathbf{y}_k$ we can write $\phi(\mathbf{x})$ in terms of $\mathbf{y}_k$ and a couple of other terms:

$$
\begin{align}
\phi(\mathbf{x}) &= \phi(\mathbf{y}_k + \mathbf{x} - \mathbf{y}_k)\nonumber\\

&= \frac{1}{2}(\mathbf{y}_k + \mathbf{x} - \mathbf{y}_k)^\text{T}\mathbf{A}(\mathbf{y}_k + \mathbf{x} - \mathbf{y}_k) - (\mathbf{y}_k + \mathbf{x} - \mathbf{y}_k)^\text{T}\mathbf{b}\\

&= \phi(\mathbf{y}_k) + \frac{1}{2}||\mathbf{x} - \mathbf{y}_k||_\mathbf{A}^2 - (\mathbf{x} - \mathbf{y}_k)^\text{T}\mathbf{r}(\mathbf{y}_k).\quad(\text{let } \mathbf{x} = \mathbf{x}_k = \mathbf{x}_{k - 1} + \alpha_k\mathbf{p}_k \text{ and } \mathbf{y}_k = \mathbf{x}_{k - 1})\\\\



\phi(\mathbf{x}_k) &= \phi(\mathbf{x}_{k-1}) + \frac{\alpha_k^2}{2}||\mathbf{p}_k||_\mathbf{A}^2 - \alpha_k\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\\

&= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\alpha_k^2 - 2\alpha_k\frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)\\

&= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\left(\alpha_k - \frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)^2 - \frac{(\mathbf{p}_k^\text{T}\mathbf{r}_{k-1})^2}{||\mathbf{p}_k||_\mathbf{A}^4}\right)\\

&= \phi(\mathbf{x}_{k-1}) + \frac{||\mathbf{p}_k||_\mathbf{A}^2}{2}\left(\alpha_k - \frac{\mathbf{p}_k^\text{T}\mathbf{r}_{k-1}}{||\mathbf{p}_k||_\mathbf{A}^2}\right)^2 - \frac{(\mathbf{p}_k^\text{T}\mathbf{r}_{k-1})^2}{2||\mathbf{p}_k||_\mathbf{A}^2}

\end{align}
$$

What this result shows is that regardless of the search direction $\mathbf{p}_k$, we can choose $\alpha_k$ to ensure the cost function $\phi$ is being minimised by setting $\alpha_k = \mathbf{p}_k^\text{T}\mathbf{r}_{k-1}\ /\ ||\mathbf{p}_k||_\mathbf{A}^2$. In the case of steepest descent, we set $\mathbf{p}_k = \mathbf{r}_{k-1}$. An interesting application of the above provides a lower bound of the convergence rate.

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

We see that the $\mathbf{A}$-norm of $\mathbf{e}_k$ shows convergence with an upper bound based on the eigenvalues of $\mathbf{A}$. We conveniently chose $\alpha$ to help deduce the convergence bound so we know that by selecting the optimal $\alpha$, we have at least better convergence as stated.


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