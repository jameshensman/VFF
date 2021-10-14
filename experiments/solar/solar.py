# Copyright 2016 James Hensman
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
import GPflow
import VFF

plt.ion()
# import matplotlib2tikz
plt.close("all")

data = np.genfromtxt("solar_data.txt", delimiter=",")
X = data[:, 0:1]
Y = data[:, 2:3]
Y = (Y - Y.mean()) / Y.std()

# remove some chunks of data
X_test, Y_test = [], []

intervals = ((1620, 1650), (1700, 1720), (1780, 1800), (1850, 1870), (1930, 1950))
for low, up in intervals:
    ind = np.logical_and(X.flatten() > low, X.flatten() < up)
    X_test.append(X[ind])
    Y_test.append(Y[ind])
    X = np.delete(X, np.where(ind)[0], axis=0)
    Y = np.delete(Y, np.where(ind)[0], axis=0)
X_test, Y_test = np.vstack(X_test), np.vstack(Y_test)


def plot(m, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    xtest = np.linspace(m.X.value.min(), m.X.value.max(), 300)[:, None]
    mu, var = m.predict_y(xtest)
    (line,) = ax.plot(xtest, mu, lw=1.5)
    ax.plot(xtest, mu + 2 * np.sqrt(var), color=line.get_color())
    ax.plot(xtest, mu - 2 * np.sqrt(var), color=line.get_color())
    ax.plot(m.X.value, m.Y.value, "r.")
    ax.plot(X_test, Y_test, "g.")
    for i in intervals:
        ax.plot([i[0], i[0]], [-2, 3], "b--")
        ax.plot([i[1], i[1]], [-2, 3], "b--")
    ax.set_ylim(-2, 3)
    ax.set_xlim(m.X.value.min(), m.X.value.max())


fig, axes = plt.subplots(5, 1, figsize=(6, 10), sharex=True)
axes = axes.flatten()

# Titsias
Z = np.linspace(X.min(), X.max(), 50)[:, None]
m = GPflow.sgpr.SGPR(X, Y, kern=GPflow.kernels.Matern52(1, lengthscales=10.0), Z=Z)
m.optimize()
plot(m, axes[0])
axes[0].set_title("(a) Variational inducing points")
axes[0].plot(m.Z.value, m.Z.value * 0, "k|")

# RFF 500
m = VFF.SSGP(X, Y, kern=GPflow.kernels.Matern52(1, lengthscales=10), num_basis=500)
m.omega.fixed = True
m.optimize()
plot(m, axes[1])
axes[1].set_title("(b) Random Fourier Features")

# SSGP
m = VFF.SSGP(X, Y, kern=GPflow.kernels.Matern52(1, lengthscales=10), num_basis=50)
m.optimize()
plot(m, axes[2])
axes[2].set_title("(c) Sparse Spectrum GP")

# VFF
m = VFF.gpr.GPR_1d(
    X, Y, np.arange(50), a=1590, b=2010, kern=GPflow.kernels.Matern52(1, lengthscales=10.0)
)
m.optimize()
plot(m, axes[3])
axes[3].set_title("(d) Variational Fourier Features")

# Full
m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.Matern52(1, lengthscales=10))
m.optimize()
plot(m, axes[4])
axes[4].set_title("(e) Full GP")
