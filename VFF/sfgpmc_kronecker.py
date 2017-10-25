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


from __future__ import print_function, absolute_import
from functools import reduce
import numpy as np
import gpflow
import tensorflow as tf
from .spectral_covariance import make_Kuu, make_Kuf
from .kronecker_ops import kvs_dot_vec


def kron_vec_sqrt_transpose(K, vec):
    """
    K is a list of objects to be kroneckered
    vec is a N x 1 tf_array
    """
    N_by_1 = tf.stack([-1, 1])

    def f(v, k):
        v = tf.reshape(v, tf.stack([k.sqrt_dims, -1]))
        v = k.matmul_sqrt_transpose(v)
        return tf.reshape(tf.transpose(v), N_by_1)  # transposing first flattens the vector in column order
    return reduce(f, K, vec)


class SFGPMC_kron(gpflow.model.GPModel):
    def __init__(self, X, Y, ms, a, b, kerns, likelihood):
        """
        X is a np array of stimuli
        Y is a np array of responses
        ms is a np integer array defining the frequencies (usually np.arange(M))
        a is a np array of the lower limits
        b is a np array of the upper limits
        kerns is a list of (Matern) kernels, one for each column of X
        likelihood is a gpflow likelihood

        # Note: we use the same frequencies for each dimension in this code for simplicity.
        """
        assert a.size == b.size == len(kerns) == X.shape[1]
        for kern in kerns:
            assert isinstance(kern, (gpflow.kernels.Matern12,
                                     gpflow.kernels.Matern32,
                                     gpflow.kernels.Matern52))
        mf = gpflow.mean_functions.Zero()
        gpflow.model.GPModel.__init__(self, X, Y, kern=None,
                                      likelihood=likelihood, mean_function=mf)
        self.num_data = X.shape[0]
        self.num_latent = 1  # multiple columns not supported in this version
        self.a = a
        self.b = b
        self.ms = ms

        # initialize variational parameters
        self.Ms = []
        for kern in kerns:
            Ncos_d = self.ms.size
            Nsin_d = self.ms.size - 1
            if isinstance(kern, gpflow.kernels.Matern12):
                Ncos_d += 1
            elif isinstance(kern, gpflow.kernels.Matern32):
                Ncos_d += 1
                Nsin_d += 1
            elif isinstance(kern, gpflow.kernels.Matern32):
                Ncos_d += 2
                Nsin_d += 1
            else:
                raise NotImplementedError
            self.Ms.append(Ncos_d + Nsin_d)

        self.kerns = gpflow.param.ParamList(kerns)

        self.V = gpflow.param.Param(np.zeros((np.prod(self.Ms), 1)))
        self.V.prior = gpflow.priors.Gaussian(0., 1.)

    def build_predict(self, X, full_cov=False):
        Kuf = [make_Kuf(k, X[:, i:i+1], a, b, self.ms) for i, (k, a, b) in enumerate(zip(self.kerns, self.a, self.b))]
        Kuu = [make_Kuu(k, a, b, self.ms) for k, a, b, in zip(self.kerns, self.a, self.b)]

        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]
        RV = kron_vec_sqrt_transpose(Kuu, self.V)  # M x 1
        mu = kvs_dot_vec([tf.transpose(KiKuf_d) for KiKuf_d in KiKuf], RV)  # N x 1
        if full_cov:
            raise NotImplementedError

        else:
            # Kff:
            var = reduce(tf.mul, [k.Kdiag(X[:, i:i+1]) for i, k in enumerate(self.kerns)])

            # Qff
            var = var - reduce(tf.mul, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)])

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def build_likelihood(self):
        Kuf = [make_Kuf(k, self.X[:, i:i+1], a, b, self.ms)
               for i, (k, a, b) in enumerate(zip(self.kerns, self.a, self.b))]
        Kuu = [make_Kuu(k, a, b, self.ms) for k, a, b, in zip(self.kerns, self.a, self.b)]

        # get mu and var of F
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]
        RV = kron_vec_sqrt_transpose(Kuu, self.V)  # M x 1
        mu = kvs_dot_vec([tf.transpose(KiKuf_d) for KiKuf_d in KiKuf], RV)  # N x 1
        var = reduce(tf.mul, [k.Kdiag(self.X[:, i:i+1]) for i, k in enumerate(self.kerns)])
        var = var - reduce(tf.mul, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)])
        var = tf.reshape(var, (-1, 1))

        E_lik = self.likelihood.variational_expectations(mu, var, self.Y)
        return tf.reduce_sum(E_lik)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    np.random.seed(0)
    X = np.random.rand(500, 2) * 2 - 1
    Y = np.cos(3*X[:, 0:1]) + 2*np.sin(5*X[:, 1:] * X[:, 0:1]) + np.random.randn(X.shape[0], 1)*0.8
    Y = np.exp(Y)

    plt.ion()

    def plot(m):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        xtest, ytest = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]
        Xtest = np.vstack((xtest.flatten(), ytest.flatten())).T
        ax1.scatter(m.X[:, 0], m.X[:, 1], 30, m.Y[:, 0],
                    vmin=0, vmax=2.7,
                    cmap=plt.cm.viridis, linewidth=0.2)
        mu, var = m.predict_f(Xtest)
        ax1.contour(xtest, ytest, mu.reshape(100, 100),
                    cmap=plt.cm.viridis, linewidths=6,
                    vmin=0, vmax=2.7)
        ax2.contour(xtest, ytest, var.reshape(100, 100),
                    cmap=plt.cm.viridis, linewidths=6,
                    vmin=0, vmax=0.5)

        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)

    lik = gpflow.likelihoods.Exponential
    for k in [gpflow.kernels.Matern32]:

        a = X.min(0) - 1.5
        b = X.max(0) + 1.5

        Ms = np.arange(10)

        m = SFGPMC_kron(X, Y, Ms, a=a, b=b, kerns=[k(1), k(1)], likelihood=lik())
        m0 = gpflow.gpmc.GPMC(X, Y, kern=k(1, active_dims=[0]) * k(1, active_dims=[1]), likelihood=lik())
        # m.kern.matern32_1

        # fix the kernels
        for k in m.kerns:
            k.lengthscales.fixed = True
            k.variance.fixed = True
        m0.kern.matern32_1.variance.fixed = True
        m0.kern.matern32_1.lengthscales.fixed = True
        m0.kern.matern32_2.variance.fixed = True
        m0.kern.matern32_2.lengthscales.fixed = True

        m.optimize()
        m0.optimize()
        plot(m)
        plot(m0)
        print(m)
