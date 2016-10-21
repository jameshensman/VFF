from __future__ import print_function, absolute_import
import numpy as np
import GPflow
import tensorflow as tf
from matplotlib import pyplot as plt
from .spectral_covariance import make_Kuu, make_Kuf
from GPflow import settings
float_type = settings.dtypes.float_type


class SFVGP_additive(GPflow.model.GPModel):
    def __init__(self, X, Y, ms, a, b, kerns, likelihood):
        """
        Here we assume the interval is [a,b]
        """
        assert a.size == b.size == len(kerns) == X.shape[1]
        for kern in kerns:
            assert isinstance(kern, (GPflow.kernels.Matern12,
                                     GPflow.kernels.Matern32,
                                     GPflow.kernels.Matern52))
        mf = GPflow.mean_functions.Zero()
        GPflow.model.GPModel.__init__(self, X, Y, kern=None,
                                      likelihood=likelihood, mean_function=mf)
        self.num_latent = 1  # multiple columns not supported in this version
        self.a = a
        self.b = b
        self.ms = ms
        self.input_dim = X.shape[1]

        # initialize variational parameters
        Ms = []
        for kern in kerns:
            Ncos_d = self.ms.size
            Nsin_d = self.ms.size - 1
            if isinstance(kern, GPflow.kernels.Matern12):
                Ncos_d += 1
            elif isinstance(kern, GPflow.kernels.Matern32):
                Ncos_d += 1
                Nsin_d += 1
            else:
                raise NotImplementedError
            Ms.append(Ncos_d + Nsin_d)

        self.kerns = GPflow.param.ParamList(kerns)

        self.q_mu = GPflow.param.Param(np.zeros((np.sum(Ms), 1)))
        pos = GPflow.transforms.positive
        self.q_sqrt = GPflow.param.ParamList([GPflow.param.Param(np.ones(M), pos) for M in Ms])

    def build_predict(self, X, full_cov=False):
        # given self.q(v), compute q(f)

        Kuf = [make_Kuf(X[:, i:i+1], a, b, self.ms) for i, (a, b) in enumerate(zip(self.a, self.b))]
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kerns, self.a, self.b)]
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]

        RKiKuf = [Kuu_d.matmul_sqrt(KiKuf_d) for Kuu_d, KiKuf_d in zip(Kuu, KiKuf)]
        KfuKiR = [tf.transpose(RKiKuf_d) for RKiKuf_d in RKiKuf]

        mu_d = [tf.matmul(KfuKiR_d, q_mu_d) for KfuKiR_d, q_mu_d in zip(KfuKiR, tf.split(0,self.input_dim,self.q_mu))]

        mu = reduce(lambda a, b: a+b, mu_d)

        tmp1 = [tf.expand_dims(q_sqrt_d, 1) * RKiKuf_d for q_sqrt_d, RKiKuf_d in zip(self.q_sqrt, RKiKuf)]
        if full_cov:
             raise NotImplementedError
        else:
            # Kff:
            var = reduce(lambda a, b: a+b, [kern.Kdiag(X[:, i:i+1]) for i, kern in enumerate(self.kerns)])

            # Projected variance Kfu Ki [A + WWT] Ki Kuf
            var = var + reduce(lambda a, b: a+b, [tf.reduce_sum(tf.square(tmp1_d), 0) for tmp1_d in tmp1])

            # Qff
            var = var - reduce(lambda a, b: a+b, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)])

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def build_KL(self):
        """
        We're working in a 'whitened' representation, so this is the KL between
        q(u) and N(0, 1)
        """
        KL = 0.5*tf.reduce_sum(tf.square(self.q_mu))  # Mahalanobis term
        KL += -0.5*tf.cast(tf.size(self.q_mu), float_type)  # Constant term.
        KL += -0.5*reduce(lambda a, b: a+b, [tf.reduce_sum(tf.log(tf.square(q_sqrt_d))) for q_sqrt_d in self.q_sqrt])  # Log det
        KL += 0.5*reduce(lambda a, b: a+b, [tf.reduce_sum(tf.square(q_sqrt_d)) for q_sqrt_d in self.q_sqrt])  # Trace term

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
        m = SFVGP(X, Y, np.arange(1000), a=-6, b=6,
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
