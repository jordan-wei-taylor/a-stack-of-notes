from scipy import stats


def gaussian_log_likelihood(w, b, X, y, noise):
    if w.ndim == 2:
        if w.shape[1] == X.shape[1]:
            w = w.T
        return stats.norm(y.reshape(-1, 1), noise).logpdf(X @ w + b).sum(axis = 0)
    return stats.norm(y, noise).logpdf(X @ w + b).sum()

