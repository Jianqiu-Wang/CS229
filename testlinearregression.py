#!/usr/bin/python3
# author: Jianqiu Wang
# email : jw2329@cornell.edu
# date  : Sep 9, 2018
# test linear regression algos

import numpy as np
import matplotlib.pyplot as plt
from linearregression import run_batch_grad_descent, run_stochastic_grad_descent


def generate_data():
    X = np.random.rand(20, 10)
    theta = np.random.randint(1, 100, 10)
    y = np.dot(X, theta) + np.random.normal(2, 100, 20)
    return (X, theta, y)


def run_test(alpha, tolerance, MAX_ITER):
    (X, theta, y) = generate_data()
    (hist_err1, theta1) = run_batch_grad_descent(X, y, alpha, MAX_ITER, tolerance)
    (hist_err2, theta2) = run_stochastic_grad_descent(X, y, alpha, MAX_ITER, tolerance)
    plt.plot(hist_err1)
    plt.plot(hist_err2)
    plt.show()


if __name__ == "__main__":
    run_test(alpha=0.01, tolerance=1e-10, MAX_ITER=100)
