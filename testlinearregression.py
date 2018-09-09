#!/usr/bin/python3
# author: Jianqiu Wang
# email : jw2329@cornell.edu
# date  : Sep 9, 2018
# test linear regression algos

import numpy as np
import matplotlib.pyplot as plt
from linearregression import run_batch_grad_descent

def generate_data():
    X = np.random.rand(100, 10)
    theta = np.random.randint(1,100,10)
    y = np.dot(X, theta) + np.random.normal(0, 1, 100)
    return (X, theta, y)

def test_batch_grad_descent():
    (X, theta, y) = generate_data()
    run_batch_grad_descent(X,y)


if __name__ == "__main__":
    theta0 = test_batch_grad_descent()
    print(theta0)
