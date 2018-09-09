#!/usr/bin/python3
# author: Jianqiu Wang
# email : jw2329@cornell.edu
# date  : Sep 9, 2018
# This python file implements Batch Gradient Descent and Stochastic Gradent
# Descent(SGD) algorithms to solve ordinary least square problems(OLS)
# Input: X: m-by-n matrix, m is the number of samples, n is the # of features
#        y: size m column vector
# Output: theta: size n column vector
# Target: find theta s.t. y = theta' * x, with minimum cost function:
#  J(theta) = 1/2 sum_i_m (h(x_i)-y_i)^2,
# where h(x_i) = theta' * x_i

import numpy as np
import matplotlib.pyplot as plt

def run_batch_grad_descent(X, y, alpha=0.1):
    """
    :param X:
    :param y:
    :param theta0:
    :return:
    """
    (m, n) = np.shape(X)
    theta = np.zeros([n, 1])
    theta0 = np.zeros([n, 1])
    err = 1
    hist_err = []
    while err > 1e-5:
        for j in range(n):
            """
            print((y - np.dot(X, theta0).reshape(-1,1)).reshape(1,-1))
            print(np.array(X[:, j]))
            theta[j] = theta0[j] + alpha * np.inner((y - np.dot(X, theta0).reshape(-1,1)).reshape(1,-1),
                                                    np.array(X[:, j]))
            """
            # loop by sample
            theta1 = theta.copy()
            for i in range(m):
                h = np.inner(theta1.reshape(-1), X[i])  # theta keeps the same for every single j
                theta[j] = theta0[j] + alpha * (y[i] - h) * X[i,j]
        # update square error
        err = sum((theta - theta0) ** 2)
        theta0 = theta.copy()
        hist_err.append(err)

    plt.plot(hist_err)
    plt.show()
    return theta