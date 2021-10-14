# Copyright 2021 ST John
# Copyright 2016 James Hensman, Nicolas Durrande
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans
from VFF.vgp import VGP_kron
import gpflow

np.random.seed(0)

X = np.loadtxt("banana_X_train", delimiter=",")
Y = np.loadtxt("banana_Y_train")[:, None]

lik = gpflow.likelihoods.Bernoulli
k = gpflow.kernels.Matern32

a = X.min(0) - 2.5
b = X.max(0) + 2.5


def plot(m, ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xtest, ytest = np.mgrid[-2.5:2.5:100j, -2.5:2.5:100j]
    Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
    for i, mark in [[0, "x"], [1, "o"]]:
        ind = Y[:, 0] == i
        ax.plot(X[ind, 0], X[ind, 1], mark)
    mu, var = m.predict_y(Xtest)
    ax.contour(xtest, ytest, mu.numpy().reshape(100, 100), levels=[0.5], colors="C0", linewidths=4)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)


# Variational Fourier Features
models = []
labels = []
for M in [2, 4, 8, 16]:
    m = VGP_kron(
        (X, Y), np.arange(M), a=a, b=b, kernels=[k(), k()], likelihood=lik(), use_two_krons=True
    )
    models.append(m)
    labels.append(f"VFF {M}")

# Pseudo-inputs
for M in [4, 8, 16, 32]:
    kern = k(active_dims=[0]) * k(active_dims=[1])
    Z, _ = kmeans(X, M)
    m = gpflow.models.SVGP(kernel=kern, likelihood=lik(), inducing_variable=Z)
    models.append(m)
    labels.append(f"IIP {M}")

# full
m = gpflow.models.VGP((X, Y), kernel=k(active_dims=[0]) * k(active_dims=[1]), likelihood=lik())
models.append(m)
labels.append("Full")

###############
for label, m in zip(labels, models):
    print(f"Optimizing {name}")
    try:
        o = gpflow.optimizers.Scipy()
        if isinstance(m, gpflow.models.ExternalDataTrainingLossMixin):
            loss = m.training_loss_closure((X, Y))
        else:
            loss = m.training_loss
        o.minimize(loss, m.trainable_variables)
    except:
        print("model optimization failed")

###############
f, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
axes = axes.flatten()
for ax, label, m in zip(axes, labels, models):
    plot(m, ax)
    ax.set_title(label)
    if isinstance(m, gpflow.models.SVGP):
        ax.plot(m.inducing_variable.Z[:, 0], m.inducing_variable.Z[:, 1], "ko", ms=4)


# m2t.save('banana_compare.tikz')
plt.savefig("banana_compare.png")
