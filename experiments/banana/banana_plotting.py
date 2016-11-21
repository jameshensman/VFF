import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans
import VFF
import GPflow

X = np.loadtxt('banana_X_train', delimiter=',')
Y = np.loadtxt('banana_Y_train')[:, None]

lik = GPflow.likelihoods.Bernoulli
k = GPflow.kernels.Matern32

a = X.min(0) - 2.5
b = X.max(0) + 2.5


def plot(m, ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xtest, ytest = np.mgrid[-2.5:2.5:100j, -2.5:2.5:100j]
    Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
    for i, mark in [[1, 'x'], [2, 'o']]:
        ind = m.Y.value[:, 0] == i
        ax.plot(m.X.value[ind, 0], m.X.value[ind, 1], mark)
    mu, var = m.predict_y(Xtest)
    ax.contour(xtest, ytest, mu.reshape(100, 100), levels=[0.5],
               colors='b', linewidths=4)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)


# Variational Fourier Features
models = []
for M in [2, 4, 8, 16]:
    m = VFF.vgp.VGP_kron(X, Y, np.arange(M), a=a, b=b, kerns=[k(1), k(1)], likelihood=lik(), use_two_krons=True)
    models.append(m)

# Pseudo-inputs
for M in [4, 8, 16, 32]:
    kern = k(1, active_dims=[0]) * k(1, active_dims=[1])
    Z, _ = kmeans(X, M)
    m = GPflow.svgp.SVGP(X, Y, kern=kern, likelihood=lik(), Z=Z)
    models.append(m)

# full
m = GPflow.vgp.VGP(X, Y, kern=k(1, active_dims=[0]) * k(1, active_dims=[1]), likelihood=lik())
models.append(m)

###############
[m.optimize() for m in models]

###############
labels = ['VFF 2', 'VFF 4', 'VFF 8', 'VFF 16', 'IIP 4', 'IIP 8', 'IIP 16', 'IIP 32', 'Full']
f, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
axes = axes.flatten()
for ax, lab, m in zip(axes, labels, models):
    plot(m, ax)
    ax.set_title(lab)
    if hasattr(m, 'Z'):
        ax.plot(m.Z.value[:, 0], m.Z.value[:, 1], 'ko', ms=4)


# m2t.save('banana_compare.tikz')
# plt.savefig('banana_compare.png')
