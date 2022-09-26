[A Stack of Notes](../a-stack-of-notes) / [Pólya Gamma Primer](polya-gamma-primer)


# Pólya-Gamma Primer

#pólya-gamma

<br>

## Bernoulli Liklihood

Consider the task of [Logistic Regresson](logistic-regression) where we have a single response target i.e. belonging or not belonging to a class. We would assume a Bernoulli likelihood with parameter $p_i$ which has the form

$$
\begin{flalign}
	&& \mathcal{L}(\mathbf{p}) \propto \prod_{i = 1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}, &&(1)
\end{flalign}
$$

where $\mathbf{p}\in[0,1]^n$, $\mathbf{y}\in\{0,\ 1\}$, and $n$ being the number of data points. We further assume that the log-odds is some linear function with our design matrix

$$
\begin{align}
	\log \frac{p_i}{1 - p_i} &= \beta_0 + \beta_1x_{i,1} + \beta_2x_{i2} + ... + \beta_mx_{i,m} = \boldsymbol{\beta}\mathbf{x}_i\tag{2}\\
	\Rightarrow p_i &= \frac{\exp(\boldsymbol{\beta}\mathbf{x}_i)}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}\tag{3}
\end{align}
$$

where $\mathbf{X}\in\mathbb{R}^{n,m}$, $\boldsymbol{\beta}\in\mathbb{R}^{m + 1}$, with $m$ being the number of features in our design matrix. We can now re-formulate our likelihood in terms of our linear coefficents resulting in

$$
\begin{align}
	\mathcal{L}(\boldsymbol{\beta}) &\propto \prod_{i = 1}^n \bigg(\frac{\exp(\boldsymbol{\beta}\mathbf{x}_i))}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i))}\bigg)^{y_i}\bigg(1 - \frac{\exp(\boldsymbol{\beta}\mathbf{x}_i))}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}\bigg)^{1 - y_i}\\
	&= \prod_{i = 1}^n \bigg(\frac{\exp(\boldsymbol{\beta}\mathbf{x}_i))}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}\bigg)^{y_i} \bigg(\frac{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)} - \frac{\exp(\boldsymbol{\beta}\mathbf{x}_i))}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}\bigg)^{1 - y_i}\\
	&= \prod_{i = 1}^n \bigg(\frac{\exp(\boldsymbol{\beta}\mathbf{x}_i))}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}\bigg)^{y_i} \bigg(\frac{1}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)}\bigg)^{1 - y_i}\\
	&= \prod_{i = 1}^n \frac{\exp(\boldsymbol{\beta}\mathbf{x}_i)^{y_i}}{1 + \exp(\boldsymbol{\beta}\mathbf{x}_i)} \tag{4}
\end{align}
$$

^bernoulli-likelihood

Due to the above expression, Bayesian inference for [Logistic Regression](logistic-regression) is intractable [(Bishop, 2006)](#^bishop).  This is because computing the evidence term would require normalising the product of a (Gaussian) prior on $\boldsymbol{\beta}$ times by the likelihood function in [Eq. (4)](#^bernoulli-likelihood). However, [Polson et al, (2013)](#^polson) introduced a new method called *Pólya-Gamma Augmentation* that allows for the construction of simple [Gibbs](gibbs-sampling) samplers for these models.

<br><br>

## Pólya-Gamma Random Variables

Let $\omega$ be a Pólya-Gamma distributed variable denoted by $\omega\sim\text{PG}(b,\ c)$ for $b > 0$ and $c\in\mathbb{R}$, then the probability density function for $\omega$ is defined as

$$
\begin{equation}
	p(\omega\ |\ b,\ c) = \frac{1}{2\pi^2} \sum_{k = 1}^\infty \frac{g_k}{(k - 1 / 2)^2 +c^2 / (4\pi^2)}\tag{5}
\end{equation}
$$

where $g_k\overset{iid}{\sim}\text{Gamma}(b,\ 1)$. Further, the author shows the property

$$
\begin{equation}
	p(\omega\mid b,\ c) = \frac{\exp(-\omega c^2 / 2)p(\omega\mid b,\ 0)}{\mathbb{E}_{p(\omega\mid b,\ 0)}[\exp(-\omega c^2 / 2)]} \tag{6}
\end{equation}
$$


[Polson et al, (2013)](#^polson) shows that all finite moments of $\omega$ can be writen in closed form. For example, the mean can be calculated immediately

$$
\begin{equation}
	\mathbb{E}[\omega\mid b,\ c] = \frac{b}{2c} \tanh (c/2).\tag{7}
\end{equation}
$$

In particular, [Polson et al, (2013)](#^polson) proved two useful properties of Pólya-Gamma random variables. First,

$$
\begin{equation}
	\frac{\exp(\psi)^a}{\big(1 + \exp(\psi)\big)^b} = 2^{-b}\exp(\kappa\psi)\int_0^\infty\exp(-\omega\psi^2/2)p(\omega\mid b,\ 0)\text{d}\omega \tag{8}
\end{equation}
$$

^polya-gamma-first

where $\kappa = a - b/2$. And second, by conditioning on $\psi$, we can normalise [Eq. (8)](#^polya-gamma-first) in $\omega$ i.e

$$
\begin{align}
	p(\omega\mid b,\ \psi) &= \frac{\exp(-\omega\psi^2/2)p(\omega\mid b,\ 0)}{\int_0^\infty\exp(-\omega\psi^2/2)p(\omega\mid b,\ 0)\text{d}\omega},\\\\
	&= \frac{\exp(-\omega\psi^2/2)p(\omega\mid b,\ 0)}{\mathbb{E}_{p(\omega\mid b,\ 0)}[\exp(-\omega \psi^2 / 2)]},\\\\
	\Rightarrow p(\omega\mid b,\ \psi) &\sim \text{PG}(b,\ \psi).\tag{9}
\end{align}
$$

---

1.  Bishop, C. M. (2006). Pattern Recognition and Machine Learning
 ^bishop
 
2. Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for logistic models using Pólya–Gamma latent variables.  Journal of the American Statistical Association, 108(504), 1339–1349. ^polson
