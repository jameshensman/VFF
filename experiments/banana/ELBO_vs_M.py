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
import tensorflow as tf
import VFF
import GPflow

X = np.loadtxt("banana_X_train", delimiter=",")
Y = np.loadtxt("banana_Y_train")[:, None]

lik = GPflow.likelihoods.Bernoulli
k = GPflow.kernels.Matern32

a = X.min(0) - 2.5
b = X.max(0) + 2.5
MM = np.arange(1, 16) * 2
num_repeats = 5


def plot(m, ax):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xtest, ytest = np.mgrid[-2.5:2.5:100j, -2.5:2.5:100j]
    Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
    for i, mark in [[0, "x"], [1, "o"]]:
        ind = m.Y.value[:, 0] == i
        ax.plot(m.X.value[ind, 0], m.X.value[ind, 1], mark)
    mu, var = m.predict_y(Xtest)
    ax.contour(xtest, ytest, mu.reshape(100, 100), levels=[0.5], colors="k", linewidths=4)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)


def build_model(M, use_two_krons):
    m = VFF.vgp.VGP_kron(
        X,
        Y,
        np.arange(M),
        a=a,
        b=b,
        kerns=[k(1), k(1)],
        likelihood=lik(),
        use_two_krons=use_two_krons,
    )
    m.kerns.fixed = True
    return m


def randomize_and_optimize(m, attempts=0, max_attempts=5):
    m.set_state(np.random.randn(m.get_free_state().size))

    try:
        m.optimize(maxiter=10000, disp=True)
        m.optimize(maxiter=10000, disp=True)  # continue in case optimize stops prematurely
    except tf.errors.InvalidArgumentError:
        if attempts >= max_attempts:
            print("Warning: model optimization failed")
        else:
            randomize_and_optimize(m, attempts + 1, max_attempts)


def run_experiment(name, use_two_krons):
    tf.reset_default_graph()
    LL = np.zeros((MM.size, num_repeats))

    # f, axes = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(15, 25))

    for i, M in enumerate(MM):
        m = build_model(M, use_two_krons=use_two_krons)
        for j in range(num_repeats):
            randomize_and_optimize(m)
            LL[i, j] = -m._objective(m.get_free_state())[0]
            print("{}, M = {}, repeat = {}, LL = {}".format(name, M, j, LL[i, j]))
        # plot(m, axes.flatten()[i])
        # axes.flatten()[i].set_title(str(M) + ', ' + str(LL[i, j]))

    # plt.savefig('plot_models_{}.png'.format(name))

    np.savetxt("banana_ELBO_{}.txt".format(name), LL)

    LLmax = np.where(np.isnan(LL), -np.inf, LL).max(1)
    f, axes = plt.subplots(1, 1)
    axes.plot(MM, LLmax, "kx")
    plt.savefig("plot_LLvsM_{}.png".format(name))


if __name__ == "__main__":
    run_experiment("doublekron", True)
    run_experiment("kron", False)
