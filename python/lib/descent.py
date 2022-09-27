import numpy as np

def compute_alpha(alpha, p, r, A):
    if alpha == 'auto':
        return (p.T @ r) / (p.T @ A @ p)
    return alpha

def steepest_descent(A, b, x, alpha, tau, max_iters, extra = None):

    check = extra is None

    if check: extra = lambda x : 0

    H = [x]
    E = [extra(x)]

    for _ in range(max_iters):
        r  = b - A @ x
        p  = r
        a  = compute_alpha(alpha, p, r, A)
        d  = a * p
        x  = x + d
        H.append(x)
        E.append(extra(x))
        if d.T @ d < tau: break

    history = np.array(H)
    extras  = np.array(E)

    return x, history, extras

def conjugate_gradient_descent(A, b, x, tau, max_iters, extra = None):

    check = extra is None

    if check: extra = lambda x : 0

    c = 0
    p = np.zeros_like(x)
    r = b - A @ x

    H = [x]
    E = [extra(x)]
    P = []

    for _ in range(min(max_iters, len(A))):
        p  = r - c * p
        a  = (r.T @ r) / (p.T @ A @ p)
        d  = a * p
        x  = x + d
        r  = b - A @ x
        c  = (r.T @ A @ p) / (p.T @ A @ p)
        H.append(x)
        P.append(p)
        E.append(extra(x))
        if d.T @ d < tau: break

    history = np.array(H)
    P       = np.array(P)
    extras  = np.array(E)
    
    return x, history, P, extras