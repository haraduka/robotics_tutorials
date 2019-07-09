import numpy as np
import matplotlib.pyplot as plt


def local_regression(x0, X, Y, tau):
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]

    # fit model: normal equations with kernel
    xw = X.T * radial_kernel(x0, X, tau)
    beta = np.linalg.pinv(xw.dot(X)).dot(xw).dot(Y)

    # predict value
    return x0.dot(beta)

def radial_kernel(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))

n = 1000
# generate dataset
X = np.linspace(-3, 3, num=n)
Y = np.log(np.abs(X ** 2 - 1) + .5)
# jitter X
X += np.random.normal(scale=0.1, size=n)

def plot_lwr(tau):
    # prediction
    domain = np.linspace(-3, 3, num=300)
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.figure()
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, Y, alpha=.3)
    plt.plot(domain, prediction, color='r')
    plt.show()

plot_lwr(1.0)
plot_lwr(0.1)
plot_lwr(0.01)
