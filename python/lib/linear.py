from   lib.abstract import AbstractLinearRegression
from   lib.descent  import steepest_descent, conjugate_gradient_descent
from   lib.metrics  import mse


import numpy as np


class LinearRegression(AbstractLinearRegression):

    def __init__(self, fit_intercept = True):
        super().__init__(locals())

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, 1)
        w = np.linalg.solve(X.T @ X, X.T @ y)
        if self.fit_intercept:
            self.b = w[0]
            self.w = w[1:]
        else:
            self.b = 0
            self.w = w
        return self


class LinearRegressionSteepestDescent(AbstractLinearRegression):

    def __init__(self, alpha = 'auto', max_iters = 200, tau = 1e-8, fit_intercept = True):
        super().__init__(locals())

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, 1)
        if self.alpha != 'auto':
            self.alpha /= len(X)
        A = X.T @ X
        b = X.T @ y
        w = np.zeros(X.shape[1])

        L = lambda w : mse(y, X @ w)

        w, self.history, self.loss = steepest_descent(A, b, w, self.alpha, self.tau, self.max_iters, L)

        if self.fit_intercept:
            self.b = w[0]
            self.w = w[1:]
        else:
            self.b = 0
            self.w = w
        return self


class LinearRegressionConjugateGradientDescent(AbstractLinearRegression):

    def __init__(self, max_iters = 200, tau = 1e-8, fit_intercept = True):
        super().__init__(locals())

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.insert(X, 0, 1, 1)
        A = X.T @ X
        b = X.T @ y
        w = np.zeros(X.shape[1])

        L = lambda w : mse(y, X @ w)
        
        w, self.history, self.P, self.loss = conjugate_gradient_descent(A, b, w, self.tau, self.max_iters, L)

        if self.fit_intercept:
            self.b = w[0]
            self.w = w[1:]
        else:
            self.b = 0
            self.w = w
        return self
