from scipy import stats


def gaussian_log_likelihood(w, b, X, y, noise):
    if w.ndim > 1:
        return stats.norm(y.reshape(-1, 1), noise).logpdf(X @ w.T + b).sum(axis = 0)
    return stats.norm(y, noise).logpdf(X @ w + b).sum()

