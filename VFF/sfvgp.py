from __future__ import print_function, absolute_import
import numpy as np
import GPflow
import tensorflow as tf
from matplotlib import pyplot as plt
from .spectral_covariance import make_Kuu, make_Kuf
from GPflow import settings
float_type = settings.dtypes.float_type


class SFVGP_1d(GPflow.model.GPModel):
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

        self.q_mu = GPflow.param.Param(np.zeros((Ncos + Nsin, 1)))
        pos = GPflow.transforms.positive
        self.q_sqrt = GPflow.param.Param(np.ones(Ncos + Nsin), pos)

    def build_predict(self, X, full_cov=False):
        # given self.q(v), compute q(f)

        Kuf = make_Kuf(self.kern, X, self.a, self.b, self.ms)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        KiKuf = Kuu.solve(Kuf)

        mu = tf.matmul(tf.transpose(KiKuf), self.q_mu)
        tmp1 = tf.expand_dims(self.q_sqrt, 1) * KiKuf
        if full_cov:
            # Kff
            var = self.kern.K(X)

            # Projected variance Kfu Ki S Ki Kuf
            var = var + tf.matmul(tf.transpose(tmp1), tmp1)

            # Qff
            var = var - tf.matmul(tf.transpose(Kuf), KiKuf)

            var = tf.expand_dims(var, 2)

        else:
            # Kff:
            var = self.kern.Kdiag(X)

            # Projected variance Kfu Ki [A + WWT] Ki Kuf
            var = var + tf.reduce_sum(tf.square(tmp1), 0)

            # Qff
            var = var - tf.reduce_sum(Kuf * KiKuf, 0)

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def build_KL(self):
        """
        We're working in a 'whitened' representation, so this is the KL between
        q(u) and N(0, 1)
        """
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        Kim = Kuu.solve(self.q_mu)
        KL = 0.5*tf.squeeze(tf.matmul(tf.transpose(Kim), self.q_mu))  # Mahalanobis term
        KL += 0.5 * Kuu.trace_KiX(tf.diag(tf.square(tf.reshape(self.q_sqrt, [-1]))))
        KL += -0.5*tf.cast(tf.size(self.q_mu), float_type)  # Constant term.
        KL += -0.5*tf.reduce_sum(tf.log(tf.square(self.q_sqrt)))  # Log det Q
        KL += 0.5*Kuu.logdet()  # Log det P
        return KL

    def build_likelihood(self):
        # compute the mean and variance of the latent function
        f_mu, f_var = self.build_predict(self.X, full_cov=False)

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, self.Y)
        return tf.reduce_sum(E_lik) - self.build_KL()


if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.rand(80, 1)*10 - 5
    X = np.sort(X, axis=0)
    Y = np.cos(3*X) + 2*np.sin(5*X) + np.random.randn(*X.shape)*0.8
    Y = np.exp(Y)

    plt.ion()

    def plot(m, col='r'):
        plt.figure()
        xtest = np.linspace(-8, 6, 1000)[:, None]
        plt.plot(m.X, m.Y, 'kx')
        mu, var = m.predict_f(xtest)
        plt.plot(xtest, np.exp(mu), col)
        plt.plot(xtest, np.exp(mu + 2*np.sqrt(var)), col+'--')
        plt.plot(xtest, np.exp(mu - 2*np.sqrt(var)), col+'--')

    for k in [GPflow.kernels.Matern12, GPflow.kernels.Matern32]:
        m = SFVGP_1d(X, Y, np.arange(1000), a=-6, b=6,
                     kern=k(1),
                     likelihood=GPflow.likelihoods.Exponential())
        m0 = GPflow.vgp.VGP(X, Y, kern=k(1),
                            likelihood=GPflow.likelihoods.Exponential())

        m.optimize()
        m0.optimize()
        plot(m, 'r')
        plot(m0, 'b')
        print(m)
        print(m0)
