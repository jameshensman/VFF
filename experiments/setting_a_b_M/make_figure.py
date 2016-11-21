import numpy as np
from matplotlib import pyplot as plt
import GPflow
from gpr_special import GPR_1d
# from matplotlib2tikz import save as savefig
plt.ion()
plt.close('all')

np.random.seed(0)
N = 20
X = np.random.rand(N, 1)
K = GPflow.kernels.Matern32(1, lengthscales=0.2).compute_K_symm(X)
Y = np.random.multivariate_normal(np.zeros(N), K + np.eye(N)*0.05).reshape(-1, 1)


def plot(m, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    xtest = np.linspace(-2, 3, 200)[:, None]
    mu, var = m.predict_f(xtest)
    line, = ax.plot(xtest, mu, lw=1.5)
    # ax.fill_between(xtest[:, 0],
    #                 mu[:, 0] + 2*np.sqrt(var[:, 0]),
    #                 mu[:, 0] - 2*np.sqrt(var[:, 0]), color='blue', alpha=0.2)
    ax.plot(xtest[:, 0], mu[:, 0] + 2*np.sqrt(var[:, 0]), color='blue', lw=0.5)
    ax.plot(xtest[:, 0], mu[:, 0] - 2*np.sqrt(var[:, 0]), color='blue', lw=0.5)
    ax.plot(m.X.value, m.Y.value, 'kx', mew=1.5)
    ax.vlines([m.a, m.b], -5, 5, 'k', linestyle='--')
    ax.set_xlim(-2, 3)
    ax.set_title('${0:.2f}$'.format(float(m.compute_log_likelihood())))


a_s = np.array([0.2, -0.1, -0.5, -1.5])
b_s = 1 - a_s
Ms = np.array([8, 16, 32])


f, axes = plt.subplots(a_s.size, Ms.size, sharex=True, sharey=True)
for i, (a, b) in enumerate(zip(a_s, b_s)):
    for j, M in enumerate(Ms):
        m = GPR_1d(X, Y, np.arange(M), a=a, b=b, kern=GPflow.kernels.Matern32(1, lengthscales=0.2))
        m.likelihood.variance = 0.05
        # m.optimize()
        plot(m, axes[i, j])
# savefig('figure.tikz', figurewidth='\\figurewidth', figureheight='\\figureheight')
