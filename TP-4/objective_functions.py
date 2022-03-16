import numpy as np


def peaks(x):
    x = x.T
    F = (
        3 * (1 - x[0]) ** 2 * np.exp(-(x[0] ** 2) - (x[1] + 1) ** 2)
        - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 5) *
        np.exp(-x[0] ** 2 - x[1] ** 2)
        - 1 / 3 * np.exp(-((x[0] + 1) ** 2) - x[1] ** 2)
    )
    return F


def rastrigin(x):
    x = x.reshape(1, -1).T
    Q = np.eye(len(x))
    X = Q.dot(x)

    n = len(X)
    F = 0

    for i in range(n):
        F = F + X[i]**2 - 10*np.cos(2*np.pi*X[i])

    return F[0]
