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


class GPMC_1d(gpflow.models.GPModel):
    def __init__(
        self, X, Y, ms, a, b, kern, likelihood, mean_function=gpflow.mean_functions.Zero()
    ):
        """
        Here we assume the interval is [a,b]
        """
        assert X.shape[1] == 1
        assert isinstance(
            kern, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
        )
        kern = kern
        gpflow.models.GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

        # initialize variational parameters
        Ncos = self.ms.size
        Nsin = self.ms.size - 1
        if isinstance(self.kern, gpflow.kernels.Matern12):
            Ncos += 1
        elif isinstance(self.kern, gpflow.kernels.Matern32):
            Ncos += 1
            Nsin += 1
        else:
            raise NotImplementedError

        self.V = gpflow.param.Param(np.zeros((Ncos + Nsin, 1)))
        self.V.prior = gpflow.priors.Gaussian(0.0, 1.0)

    @gpflow.autoflow()
    def mats(self):
        Kuf = make_Kuf(self.X, self.a, self.b, self.ms)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        KiKuf = Kuu.solve(Kuf)
        var = self.kern.K(X)

        return var, KiKuf, Kuf

    def build_predict(self, X, full_cov=False):
        # given self.V, compute q(f)

        Kuf = make_Kuf(X, self.a, self.b, self.ms)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        KiKuf = Kuu.solve(Kuf)
        RKiKuf = Kuu.matmul_sqrt(KiKuf)
        mu = tf.matmul(tf.transpose(RKiKuf), self.V)

        if full_cov:
            # Kff
            var = self.kern.K(X)

            # Qff
            var = var - tf.matmul(tf.transpose(Kuf), KiKuf)

            var = tf.expand_dims(var, 2)

        else:
            # Kff:
            var = self.kern.Kdiag(X)

            # Qff
            var = var - tf.reduce_sum(Kuf * KiKuf, 0)

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def build_likelihood(self):
        # compute the mean and variance of the latent function
        f_mu, f_var = self.build_predict(self.X, full_cov=False)

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, self.Y)
        return tf.reduce_sum(E_lik)


def kron_vec_sqrt_transpose(K, vec):
    """
    K is a list of objects to be kroneckered
    vec is a N x 1 tf_array
    """
    N_by_1 = tf.stack([-1, 1])

    def f(v, k):
        v = tf.reshape(v, tf.stack([k.sqrt_dims, -1]))
        v = k.matmul_sqrt_transpose(v)
        return tf.reshape(
            tf.transpose(v), N_by_1
        )  # transposing first flattens the vector in column order

    return reduce(f, K, vec)


class GPMC_kron(gpflow.models.GPModel):
    def __init__(self, X, Y, ms, a, b, kerns, likelihood, mean_function=None):
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
            assert isinstance(
                kern, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
            )
        if mean_function is None:
            mean_function = gpflow.mean_functions.Zero()
        gpflow.models.GPModel.__init__(
            self, X, Y, kern=None, likelihood=likelihood, mean_function=mean_function
        )
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
            elif isinstance(kern, gpflow.kernels.Matern52):
                Ncos_d += 2
                Nsin_d += 1
            else:
                raise NotImplementedError
            self.Ms.append(Ncos_d + Nsin_d)

        self.kerns = gpflow.param.ParamList(kerns)

        self.V = gpflow.param.Param(np.zeros((np.prod(self.Ms), 1)))
        self.V.prior = gpflow.priors.Gaussian(0.0, 1.0)

    def build_predict(self, X, full_cov=False):
        Kuf = [
            make_Kuf(k, X[:, i : i + 1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kerns, self.a, self.b))
        ]
        Kuu = [make_Kuu(k, a, b, self.ms) for k, a, b, in zip(self.kerns, self.a, self.b)]

        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]
        RV = kron_vec_sqrt_transpose(Kuu, self.V)  # M x 1
        mu = kvs_dot_vec([tf.transpose(KiKuf_d) for KiKuf_d in KiKuf], RV)  # N x 1
        if full_cov:
            raise NotImplementedError

        else:
            # Kff:
            var = reduce(tf.mul, [k.Kdiag(X[:, i : i + 1]) for i, k in enumerate(self.kerns)])

            # Qff
            var = var - reduce(
                tf.mul, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)]
            )

            var = tf.reshape(var, (-1, 1))

        return mu + self.mean_function(self.X), var

    def build_likelihood(self):
        Kuf = [
            make_Kuf(k, self.X[:, i : i + 1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kerns, self.a, self.b))
        ]
        Kuu = [make_Kuu(k, a, b, self.ms) for k, a, b, in zip(self.kerns, self.a, self.b)]

        # get mu and var of F
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]
        RV = kron_vec_sqrt_transpose(Kuu, self.V)  # M x 1
        mu = kvs_dot_vec([tf.transpose(KiKuf_d) for KiKuf_d in KiKuf], RV)  # N x 1
        mu += self.mean_function(self.X)
        var = reduce(tf.mul, [k.Kdiag(self.X[:, i : i + 1]) for i, k in enumerate(self.kerns)])
        var = var - reduce(
            tf.mul, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)]
        )
        var = tf.reshape(var, (-1, 1))

        E_lik = self.likelihood.variational_expectations(mu, var, self.Y)
        return tf.reduce_sum(E_lik)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    np.random.seed(0)
    X = np.random.rand(80, 1) * 10 - 5
    X = np.sort(X, axis=0)
    Y = np.cos(3 * X) + 2 * np.sin(5 * X) + np.random.randn(*X.shape) * 0.8
    Y = np.exp(Y)

    plt.ion()

    def plot(m, samples, col="r"):
        xtest = np.linspace(-5.5, 5.5, 500)[:, None]
        plt.figure()
        for s in samples[::10]:
            m.set_state(s)
            f = m.predict_f_samples(xtest, 10).squeeze()
            # f, _ = m.predict_f(xtest)
            plt.plot(xtest.flatten(), np.exp(f.T), col, alpha=0.01)
        plt.plot(X, Y, "kx", mew=2)
        plt.ylim(0, 100)

    # for k in [gpflow.kernels.Matern12, gpflow.kernels.Matern32]:
    for k in [gpflow.kernels.Matern32]:
        m = GPMC_1d(
            X, Y, np.arange(1000), a=-6, b=6, kern=k(1), likelihood=gpflow.likelihoods.Exponential()
        )
        m0 = gpflow.gpmc.GPMC(X, Y, kern=k(1), likelihood=gpflow.likelihoods.Exponential())

        m.kern.variance = 2.5
        m.kern.variance.fixed = True
        m0.kern.variance = 2.5
        m0.kern.variance.fixed = True
        m.kern.lengthscales = 0.3
        m.kern.lengthscales.fixed = True
        m0.kern.lengthscales = 0.3
        m0.kern.lengthscales.fixed = True

        m.optimize()
        m0.optimize()

        samples = m.sample(1000, epsilon=0.11, Lmax=20, verbose=1)
        samples0 = m0.sample(1000, epsilon=0.11, Lmax=20, verbose=1)

        plot(m, samples)
        plot(m0, samples0, "b")
        print(m)
        print(m0)
