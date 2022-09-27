from   itertools import product
from   scipy     import stats

import numpy as np

def gen_linear_data(n, m, noise = 0.1, w = None, b = None, random_state = None):

    if random_state is not None:
        np.random.seed(random_state)
    
    if isinstance(m, int):
        cov = np.eye(m)
    else:
        cov = m
        m   = len(cov)

    # design matrix
    X = stats.multivariate_normal([0] * len(cov), cov).rvs(n)

    # linear co-efficients
    w = np.array(w) if w is not None else stats.uniform(-3, 6).rvs(m)
    b = np.array(b) if b is not None else stats.uniform(-3, 6)

    # true target response
    t = X @ w + b

    # noisy target reponse
    y = t + stats.norm(0, noise).rvs(n)

    # true evaluation at fixed corners
    Xc = np.array(list(product(*[[-5, 5]] * m)))
    yc = Xc @ w + b
    
    # group the data, parameters, and corner evaluations
    data    = (X, y)
    params  = (w, b)
    corners = (Xc, yc)

    return data, params, corners