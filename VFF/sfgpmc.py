from __future__ import print_function, absolute_import
import numpy as np
import GPflow
import tensorflow as tf
from matplotlib import pyplot as plt
from .spectral_covariance import make_Kuu, make_Kuf


class SFGPMC_1d(GPflow.model.GPModel):
    def __init__(self, X, Y, ms, a, b, kern, likelihood,
                 mean_function=GPflow.mean_functions.Zero()):
        """
        Here we assume the interval is [a,b]
        """
        assert X.shape[1] == 1
        assert isinstance(kern, (GPflow.kernels.Matern12,
                                 GPflow.kernels.Matern32,
                                 GPflow.kernels.Matern52))
        kern = kern
        GPflow.model.GPModel.__init__(self, X, Y, kern,
                                      likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

        # initialize variational parameters
        Ncos = self.ms.size
        Nsin = self.ms.size - 1
        if isinstance(self.kern, GPflow.kernels.Matern12):
            Ncos += 1
        elif isinstance(self.kern, GPflow.kernels.Matern32):
            Ncos += 1
            Nsin += 1
        else:
            raise NotImplementedError

        self.V = GPflow.param.Param(np.zeros((Ncos + Nsin, 1)))
        self.V.prior = GPflow.priors.Gaussian(0., 1.)

    @GPflow.model.AutoFlow()
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


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.rand(80, 1)*10 - 5
    X = np.sort(X, axis=0)
    Y = np.cos(3*X) + 2*np.sin(5*X) + np.random.randn(*X.shape)*0.8
    Y = np.exp(Y)

    plt.ion()

    def plot(m, samples, col='r'):
        xtest = np.linspace(-5.5, 5.5, 500)[:, None]
        plt.figure()
        for s in samples[::10]:
            m.set_state(s)
            f = m.predict_f_samples(xtest, 10).squeeze()
            # f, _ = m.predict_f(xtest)
            plt.plot(xtest.flatten(), np.exp(f.T), col, alpha=0.01)
        plt.plot(X, Y, 'kx', mew=2)
        plt.ylim(0, 100)

    # for k in [GPflow.kernels.Matern12, GPflow.kernels.Matern32]:
    for k in [GPflow.kernels.Matern32]:
        m = SFGPMC_1d(X, Y, np.arange(1000), a=-6, b=6,
                      kern=k(1),
                      likelihood=GPflow.likelihoods.Exponential())
        m0 = GPflow.gpmc.GPMC(X, Y, kern=k(1),
                              likelihood=GPflow.likelihoods.Exponential())

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
        plot(m0, samples0, 'b')
        print(m)
        print(m0)
