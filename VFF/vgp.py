# Copyright 2021 ST John
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
from gpflow import default_float
import tensorflow as tf
from .spectral_covariance import make_Kuu, make_Kuf, make_Kuf_np
from .kronecker_ops import kvs_dot_vec, kron_vec_apply, kvs_dot_mat, kron_mat_apply, kron


class VGP_1d(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(
        self, data, ms, a, b, kernel, likelihood, mean_function=gpflow.mean_functions.Zero()
    ):
        """
        Here we assume the interval is [a,b]
        """
        X, _ = data
        assert X.shape[1] == 1
        assert isinstance(
            kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
        )
        num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)
        self.data = data
        self.a = a
        self.b = b
        self.ms = ms

        # initialize variational parameters
        Ncos = self.ms.size
        Nsin = self.ms.size - 1

        self.q_mu = gpflow.Parameter(np.zeros((Ncos + Nsin, 1)))
        self.q_sqrt = gpflow.Parameter(np.ones(Ncos + Nsin), transform=gpflow.utilities.positive())

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov
        X = Xnew
        # given self.q(v), compute q(f)

        Kuf = make_Kuf(self.kernel, X, self.a, self.b, self.ms)
        Kuu = make_Kuu(self.kernel, self.a, self.b, self.ms)
        KiKuf = Kuu.solve(Kuf)

        mu = tf.matmul(KiKuf, self.q_mu, transpose_a=True)
        tmp1 = tf.expand_dims(self.q_sqrt, 1) * KiKuf
        if full_cov:
            # Kff
            var = self.kernel(X, full_cov=True)
            var = var + tf.matmul(
                tmp1, tmp1, transpose_a=True
            )  # Projected variance Kfu Ki S Ki Kuf
            var = var - tf.matmul(Kuf, KiKuf, transpose_a=True)  # Qff
            var = tf.expand_dims(var, 2)

        else:
            var = self.kernel(X, full_cov=False)  # Kff
            var = var + tf.reduce_sum(
                tf.square(tmp1), 0
            )  # Projected variance Kfu Ki [A + WWT] Ki Kuf
            var = var - tf.reduce_sum(Kuf * KiKuf, 0)  # Qff
            var = tf.reshape(var, (-1, 1))

        return mu, var

    def compute_KL(self):
        """
        We're working in a 'whitened' representation, so this is the KL between
        q(u) and N(0, 1)
        """
        Kuu = make_Kuu(self.kernel, self.a, self.b, self.ms)
        Kim = Kuu.solve(self.q_mu)
        KL = 0.5 * tf.squeeze(tf.matmul(Kim, self.q_mu, transpose_a=True))  # Mahalanobis term
        KL += 0.5 * Kuu.trace_KiX(tf.linalg.diag(tf.square(tf.reshape(self.q_sqrt, [-1]))))
        KL += -0.5 * tf.cast(tf.size(self.q_mu), default_float())  # Constant term.
        KL += -0.5 * tf.reduce_sum(tf.math.log(tf.square(self.q_sqrt)))  # Log det Q
        KL += 0.5 * Kuu.logdet()  # Log det P
        return KL

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        X, Y = self.data

        # compute the mean and variance of the latent function
        f_mu, f_var = self.predict_f(X, full_cov=False)

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, Y)
        return tf.reduce_sum(E_lik) - self.compute_KL()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    np.random.seed(0)
    X = np.random.rand(80, 1) * 10 - 5
    X = np.sort(X, axis=0)
    Y = np.cos(3 * X) + 2 * np.sin(5 * X) + np.random.randn(*X.shape) * 0.8
    Y = np.exp(Y)

    plt.ion()

    def plot(m, col="r"):
        plt.figure()
        xtest = np.linspace(-8, 6, 1000)[:, None]
        X, Y = m.data
        plt.plot(X, Y, "kx")
        mu, var = m.predict_f(xtest)
        plt.plot(xtest, np.exp(mu), col)
        plt.plot(xtest, np.exp(mu + 2 * np.sqrt(var)), col + "--")
        plt.plot(xtest, np.exp(mu - 2 * np.sqrt(var)), col + "--")

    for k in [gpflow.kernels.Matern12, gpflow.kernels.Matern32]:
        m = VGP_1d(
            (X, Y),
            np.arange(1000),
            a=-6,
            b=6,
            kernel=k(1),
            likelihood=gpflow.likelihoods.Exponential(),
        )
        m0 = gpflow.models.VGP((X, Y), kernel=k(1), likelihood=gpflow.likelihoods.Exponential())

        opt = gpflow.optimizers.Scipy()
        opt.minimize(m.training_loss, m.trainable_variables)
        opt.minimize(m0.training_loss, m0.trainable_variables)
        plot(m, "r")
        plot(m0, "b")
        print(m)
        print(m0)


class VGP_additive(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernels, likelihood):
        """
        Here we assume the interval is [a,b]
        """
        X, Y = data
        assert a.size == b.size == len(kernels) == X.shape[1]
        for kernel in kernels:
            assert isinstance(
                kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
            )
        assert Y.shape[1] == 1
        num_latent_gps = 1  # multiple columns not supported in this version
        super().__init__(kernel=None, likelihood=likelihood, num_latent_gps=num_latent_gps)
        self.data = data
        self.a = a
        self.b = b
        self.ms = ms
        self.input_dim = X.shape[1]

        # initialize variational parameters
        self.Ms = []
        for kernel in kernels:
            Ncos_d = self.ms.size
            Nsin_d = self.ms.size - 1
            if isinstance(kernel, gpflow.kernels.Matern12):
                Ncos_d += 1
            elif isinstance(kernel, gpflow.kernels.Matern32):
                Ncos_d += 1
                Nsin_d += 1
            else:
                raise NotImplementedError
            self.Ms.append(Ncos_d + Nsin_d)

        self.kernels = kernels

        self.q_mu = gpflow.Parameter(np.zeros((np.sum(self.Ms), 1)))
        self.q_sqrt = [
            gpflow.Parameter(np.ones(M), transform=gpflow.utilities.positive()) for M in self.Ms
        ]

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov
        X = Xnew
        # given self.q(v), compute q(f)

        Kuf = [
            make_Kuf(k, X[:, i : i + 1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))
        ]
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kernels, self.a, self.b)]
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]

        RKiKuf = [Kuu_d.matmul_sqrt(KiKuf_d) for Kuu_d, KiKuf_d in zip(Kuu, KiKuf)]
        KfuKiR = [tf.transpose(RKiKuf_d) for RKiKuf_d in RKiKuf]  # XXX

        mu_d = [
            tf.matmul(KfuKiR_d, q_mu_d)
            for KfuKiR_d, q_mu_d in zip(KfuKiR, tf.split(self.q_mu, self.Ms, axis=0))
        ]

        mu = reduce(lambda a, b: a + b, mu_d)

        tmp1 = [
            tf.expand_dims(q_sqrt_d, 1) * RKiKuf_d
            for q_sqrt_d, RKiKuf_d in zip(self.q_sqrt, RKiKuf)
        ]
        if full_cov:
            raise NotImplementedError
        else:
            # Kff:
            var = reduce(
                tf.add,
                [kernel(X[:, i : i + 1], full_cov=False) for i, kernel in enumerate(self.kernels)],
            )

            # Projected variance Kfu Ki [A + WWT] Ki Kuf
            var = var + reduce(tf.add, [tf.reduce_sum(tf.square(tmp1_d), 0) for tmp1_d in tmp1])

            # Qff
            var = var - reduce(
                tf.add, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)]
            )

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def compute_KL(self):
        """
        We're working in a 'whitened' representation, so this is the KL between
        q(u) and N(0, 1)
        """
        KL = 0.5 * tf.reduce_sum(tf.square(self.q_mu))  # Mahalanobis term
        KL += -0.5 * tf.cast(tf.size(self.q_mu), default_float())  # Constant term.
        KL += -0.5 * reduce(
            tf.add, [tf.reduce_sum(tf.math.log(tf.square(q_sqrt_d))) for q_sqrt_d in self.q_sqrt]
        )  # Log det
        KL += 0.5 * reduce(
            tf.add, [tf.reduce_sum(tf.square(q_sqrt_d)) for q_sqrt_d in self.q_sqrt]
        )  # Trace term

        return KL

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        X, Y = self.data

        # compute the mean and variance of the latent function
        f_mu, f_var = self.predict_f(X, full_cov=False)

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, Y)
        return tf.reduce_sum(E_lik) - self.compute_KL()


class VGP_kron(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernels, likelihood, use_two_krons=False, use_extra_ranks=0):
        """
        Here we assume the interval is [a,b]
        """
        X, Y = data
        assert a.size == b.size == len(kernels) == X.shape[1]
        for kernel in kernels:
            assert isinstance(
                kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
            )
        assert Y.shape[1] == 1
        num_latent_gps = 1  # multiple columns not supported in this version
        super().__init__(kernel=None, likelihood=likelihood, num_latent_gps=num_latent_gps)
        self.data = data
        self.a = a
        self.b = b
        self.ms = ms

        self.kernels = kernels

        # initialize variational parameters
        self.Ms = []
        for kernel in kernels:
            Ncos_d = self.ms.size
            Nsin_d = self.ms.size - 1
            self.Ms.append(Ncos_d + Nsin_d)

        self.q_mu = gpflow.Parameter(np.zeros((np.prod(self.Ms), 1)))

        # The covariance matrix
        self.q_sqrt_kron = [gpflow.Parameter(np.eye(M)) for M in self.Ms]
        self.use_two_krons = use_two_krons
        self.use_extra_ranks = use_extra_ranks
        assert not (
            use_extra_ranks and use_two_krons
        ), "can only use one extra covariance structure at a time!"
        if use_two_krons:
            # same as above, but with different init to break symmetry
            self.q_sqrt_kron_2 = [gpflow.Parameter(np.eye(M) + 0.01) for M in self.Ms]
        elif use_extra_ranks:
            self.q_sqrt_W = gpflow.Parameter(np.zeros((np.prod(self.Ms), use_extra_ranks)))

        # pre-compute Kuf
        self._Kuf = [
            tf.constant(make_Kuf_np(X[:, i : i + 1], ai, bi, self.ms))
            for i, (ai, bi) in enumerate(zip(self.a, self.b))
        ]

    def __getstate__(self):
        d = super().__getstate__()
        d.pop("_Kuf")
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        X, _ = self.data
        self._Kuf = [
            tf.constant(make_Kuf_np(X[:, i : i + 1], ai, bi, self.ms))
            for i, (ai, bi) in enumerate(zip(self.a, self.b))
        ]

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov
        X = Xnew
        # given self.q(v), compute q(f)

        Kuf = [
            make_Kuf(k, X[:, i : i + 1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))
        ]

        return self._predict_f(X, Kuf, full_cov)

    def _predict_f(self, X, Kuf, full_cov):
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kernels, self.a, self.b)]
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]
        KfuKi = [tf.transpose(mat) for mat in KiKuf]  # XXX

        mu = kvs_dot_vec(KfuKi, self.q_mu)

        if full_cov:
            raise NotImplementedError
        else:
            # Kff:
            var = reduce(
                tf.multiply,
                [k(X[:, i : i + 1], full_cov=False) for i, k in enumerate(self.kernels)],
            )

            # Projected variance Kfu Ki [WWT] Ki Kuf
            Ls = [tf.linalg.band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron]
            tmp = [tf.matmul(L, KiKuf_d, transpose_a=True) for L, KiKuf_d in zip(Ls, KiKuf)]
            var = var + reduce(tf.multiply, [tf.reduce_sum(tf.square(tmp_d), 0) for tmp_d in tmp])

            if self.use_two_krons:
                Ls = [tf.linalg.band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron_2]
                tmp = [tf.matmul(L, KiKuf_d, transpose_a=True) for L, KiKuf_d in zip(Ls, KiKuf)]
                var = var + reduce(
                    tf.multiply, [tf.reduce_sum(tf.square(tmp_d), 0) for tmp_d in tmp]
                )
            elif self.use_extra_ranks:
                for i in range(self.use_extra_ranks):
                    tmp = kvs_dot_vec(KfuKi, self.q_sqrt_W[:, i : i + 1])
                    var = var + tf.reduce_sum(tf.square(tmp), 1)

            # Qff
            var = var - reduce(
                tf.multiply,
                [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)],
            )

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def predict_train(self):
        X, _ = self.data
        Kuf = self._Kuf
        return self._predict_f(X, Kuf, full_cov=False)

    def compute_KL(self):
        """
        The covariance of q(u) has a kronecker structure, so
        appropriate reductions apply for the trace and logdet terms.
        """
        # Mahalanobis term, m^T K^{-1} m
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kernels, self.a, self.b)]
        Kim = kron_vec_apply(Kuu, self.q_mu, "solve")
        KL = 0.5 * tf.reduce_sum(self.q_mu * Kim)

        # Constant term
        KL += -0.5 * tf.cast(tf.size(self.q_mu), default_float())

        # Log det term
        Ls = [tf.linalg.band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron]
        N_others = [float(np.prod(self.Ms)) / M for M in self.Ms]
        Q_logdets = [tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L)))) for L in Ls]
        KL += -0.5 * reduce(tf.add, [N * logdet for N, logdet in zip(N_others, Q_logdets)])

        # trace term tr(K^{-1} Sigma_q)
        Ss = [tf.matmul(L, L, transpose_b=True) for L in Ls]
        traces = [K.trace_KiX(S) for K, S, in zip(Kuu, Ss)]
        KL += 0.5 * reduce(tf.multiply, traces)  # kron-trace is the produce of traces

        # log det term Kuu
        Kuu_logdets = [K.logdet() for K in Kuu]
        KL += 0.5 * reduce(tf.add, [N * logdet for N, logdet in zip(N_others, Kuu_logdets)])

        if self.use_two_krons:
            # extra logdet terms:
            Ls_2 = [tf.linalg.band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron_2]
            LiL = [tf.linalg.triangular_solve(L1, L2) for L1, L2 in zip(Ls, Ls_2)]
            eigvals = [
                tf.linalg.eigh(tf.matmul(mat, mat, transpose_a=True))[0] for mat in LiL
            ]  # discard eigenvectors  # XXX
            eigvals_kronned = kron([tf.reshape(e, [1, -1]) for e in eigvals])
            KL += -0.5 * tf.reduce_sum(tf.math.log(1 + eigvals_kronned))

            # extra trace terms
            Ss = [tf.matmul(L, L, transpose_b=True) for L in Ls_2]
            traces = [K.trace_KiX(S) for K, S, in zip(Kuu, Ss)]
            KL += 0.5 * reduce(tf.multiply, traces)  # kron-trace is the produce of traces

        elif self.use_extra_ranks:
            # extra logdet terms
            KiW = kron_mat_apply(Kuu, self.q_sqrt_W, "solve", self.use_extra_ranks)
            WTKiW = tf.matmul(self.q_sqrt_W, KiW, transpose_a=True)
            L_extra = tf.linalg.cholesky(np.eye(self.use_extra_ranks) + WTKiW)
            KL += -0.5 * tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L_extra))))

            # extra trace terms
            KL += 0.5 * tf.reduce_sum(tf.linalg.diag_part(WTKiW))

        return KL

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        _, Y = self.data

        # compute the mean and variance of the latent function
        f_mu, f_var = self.predict_train()

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, Y)
        return tf.reduce_sum(E_lik) - self.compute_KL()


class VGP_kron_anyvar(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(self, data, ms, a, b, kernels, likelihood):
        """
        Here we assume the interval is [a,b]
        We do *not* assume that the variance of q(u) has a kronecker structure.

        This can get very computationally heavy very quickly, use with caution!.
        """
        X, Y = data
        assert a.size == b.size == len(kernels) == X.shape[1]
        for kernel in kernels:
            assert isinstance(
                kernel, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
            )
        assert Y.shape[1] == 1
        num_latent_gps = 1  # multiple columns not supported in this version
        super().__init__(kernel=None, likelihood=likelihood, num_latent_gps=num_latent_gps)
        self.data = data
        self.a = a
        self.b = b
        self.ms = ms

        # initialize variational parameters
        Ms = []
        for kernel in kernels:
            Ncos_d = self.ms.size
            Nsin_d = self.ms.size - 1
            Ms.append(Ncos_d + Nsin_d)
        self.Ms = Ms

        self.kernels = kernels

        self.q_mu = gpflow.Parameter(np.zeros((np.prod(Ms), 1)))

        # The covariance matrix gets very big very quickly
        self.q_sqrt = gpflow.Parameter(np.eye(np.prod(Ms)))

        # pre-compute the Kuf matrices
        self._Kuf = [
            tf.constant(make_Kuf_np(X[:, i : i + 1], ai, bi, self.ms))
            for i, (ai, bi) in enumerate(zip(self.a, self.b))
        ]

    def __getstate__(self):
        d = super().__getstate__()
        d.pop("_Kuf")
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        X, _ = self.data
        self._Kuf = [
            tf.constant(make_Kuf_np(X[:, i : i + 1], ai, bi, self.ms))
            for i, (ai, bi) in enumerate(zip(self.a, self.b))
        ]

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov
        X = Xnew
        # given self.q(v), compute q(f)

        Kuf = [
            make_Kuf(k, X[:, i : i + 1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kernels, self.a, self.b))
        ]
        return self._predict_f(X, Kuf, full_cov)

    def _predict_f(self, X, Kuf, full_cov):
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kernels, self.a, self.b)]
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]
        KfuKi = [tf.transpose(mat) for mat in KiKuf]  # XXX

        mu = kvs_dot_vec(KfuKi, self.q_mu)

        L = tf.linalg.band_part(self.q_sqrt, -1, 0)
        tmp1 = kvs_dot_mat(KfuKi, L, np.prod(self.Ms))

        if full_cov:
            raise NotImplementedError
        else:
            # Kff:
            var = reduce(
                tf.multiply,
                [k(X[:, i : i + 1], full_cov=False) for i, k in enumerate(self.kernels)],
            )

            # Projected variance Kfu Ki [WWT] Ki Kuf
            # var = var + reduce(tf.multiply, [tf.reduce_sum(tf.square(tmp1_d), 0) for tmp1_d in tmp1])
            var = var + tf.reduce_sum(tf.square(tmp1), 1)

            # Qff
            var = var - reduce(
                tf.multiply,
                [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)],
            )

            var = tf.reshape(var, (-1, 1))

        return mu, var

    def predict_train(self):
        X, _ = self.data
        Kuf = self._Kuf
        return self._predict_f(X, Kuf, full_cov=False)

    def compute_KL(self):
        """
        The covariance of q(u) has a kronecker structure, so
        appropriate reductions apply for the trace and logdet terms.
        """
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kernels, self.a, self.b)]
        Kim = kron_vec_apply(Kuu, self.q_mu, "solve")
        KL = 0.5 * tf.reduce_sum(self.q_mu * Kim)  # Mahalanobis term
        KL += -0.5 * tf.cast(tf.size(self.q_mu), default_float())  # Constant term.
        L = tf.linalg.band_part(self.q_sqrt, -1, 0)
        Q_logdet = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        KL += -0.5 * Q_logdet  # log determinat Sigma_q
        S = tf.matmul(L, L, transpose_b=True)
        KL += 0.5 * tf.reduce_sum(
            tf.linalg.diag_part(kron_mat_apply(Kuu, S, "solve", np.prod(self.Ms)))
        )
        Kuu_logdets = [K.logdet() for K in Kuu]
        N_others = [tf.cast(tf.size(self.q_mu) / M, default_float()) for M in self.Ms]
        KL += 0.5 * reduce(
            tf.add, [N * logdet for N, logdet in zip(N_others, Kuu_logdets)]
        )  # kron-logdet P
        return KL

    def maximum_log_likelihood_objective(self):
        return self.elbo()

    def elbo(self):
        _, Y = self.data

        # compute the mean and variance of the latent function
        f_mu, f_var = self.predict_train()

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, Y)
        return tf.reduce_sum(E_lik) - self.compute_KL()
