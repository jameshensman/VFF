from __future__ import print_function, absolute_import
import numpy as np
import GPflow
import tensorflow as tf
from .spectral_covariance import make_Kuu, make_Kuf
from .kronecker_ops import kvs_dot_vec, make_kvs


class SFVGP_proj_kron(GPflow.model.GPModel):
    def __init__(self, latent_dim, X, Y, ms, kerns, likelihood, q_diag=False):
        assert len(kerns) == latent_dim
        for kern in kerns:
            assert isinstance(kern, (GPflow.kernels.Matern12,
                                     GPflow.kernels.Matern32,
                                     GPflow.kernels.Matern52))
        mf = GPflow.mean_functions.Zero()
        GPflow.model.GPModel.__init__(self, X, Y, kern=None,
                                      likelihood=likelihood, mean_function=mf)
        self.num_latent = 1  # multiple columns not supported in this version
        self.a = np.ones(latent_dim) * -2
        self.b = np.ones(latent_dim) * 2
        self.ms = ms

        # projection
        self.A = GPflow.param.Param(np.random.randn(X.shape[1], latent_dim))

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

        self.q_mu = GPflow.param.Param(np.zeros((np.prod(Ms), 1)))
        pos = GPflow.transforms.positive

        # The covariance matrix has a special structure. First, a Kron structured part:
        self.q_diag = q_diag
        if self.q_diag:
            self.q_sqrt_kron = GPflow.param.ParamList([GPflow.param.Param(np.ones(M), pos) for M in Ms])
        else:
            self.q_sqrt_kron = GPflow.param.ParamList([GPflow.param.Param(np.eye(M)) for M in Ms])
        # and also a low-rank part
        # rank = 2
        # self.q_sqrt_lowrank = GPflow.param.Param(np.zeros((np.prod(Ms), rank)))

    def build_predict(self, X, full_cov=False):
        # given self.q(v), compute q(f)

        X_projected = tf.nn.tanh(tf.matmul(X, self.A))
        Kuf = [make_Kuf(X_projected[:, i:i+1], a, b, self.ms) for i, (a, b) in enumerate(zip(self.a, self.b))]
        Kuu = [make_Kuu(kern, a, b, self.ms) for kern, a, b, in zip(self.kerns, self.a, self.b)]
        KiKuf = [Kuu_d.solve(Kuf_d) for Kuu_d, Kuf_d in zip(Kuu, Kuf)]

        RKiKuf = [Kuu_d.matmul_sqrt(KiKuf_d) for Kuu_d, KiKuf_d in zip(Kuu, KiKuf)]
        KfuKiR = [tf.transpose(RKiKuf_d) for RKiKuf_d in RKiKuf]

        mu = kvs_dot_vec(KfuKiR, self.q_mu)
        if self.q_diag:
            tmp1 = [tf.expand_dims(q_sqrt_d, 1) * RKiKuf_d for q_sqrt_d, RKiKuf_d in zip(self.q_sqrt_kron, RKiKuf)]
        else:
            Ls = [tf.batch_matrix_band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron]
            tmp1 = [tf.matmul(tf.transpose(L), RKiKuf_d) for L, RKiKuf_d in zip(Ls, RKiKuf)]
        # tmp2 = [kvs_dot_vec(KfuKiR, self.q_sqrt_lowrank[:, i:i+1]) for i in range(2)]

        if full_cov:
            raise NotImplementedError
        else:
            # Kff:
            var = reduce(tf.mul, [k.Kdiag(X_projected[:, i:i+1]) for i, k in enumerate(self.kerns)])

            # Projected variance Kfu Ki [A + WWT] Ki Kuf
            # kron-part
            var = var + reduce(tf.mul, [tf.reduce_sum(tf.square(tmp1_d), 0) for tmp1_d in tmp1])
            # low-rank part
            # var = var + reduce(tf.add, [tf.reshape(tf.square(tmp2_i), (-1,)) for tmp2_i in tmp2])

            # Qff
            var = var - reduce(tf.mul, [tf.reduce_sum(Kuf_d * KiKuf_d, 0) for Kuf_d, KiKuf_d in zip(Kuf, KiKuf)])

            var = tf.reshape(var, (-1, 1))

        return mu, var

    @GPflow.model.AutoFlow()
    def compute_KL(self):
        return self.build_KL()

    @GPflow.model.AutoFlow()
    def compute_KL_bits(self):

        Ls = [tf.batch_matrix_band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron]
        return [tf.reduce_sum(tf.square(L)) for L in Ls]  # kron-trace

    def build_KL(self):
        """
        We're working in a 'whitened' representation, so this is the KL between
        q(u) and N(0, 1). The covariance of q(u) has a kronecker structure, so
        appropriate reductions apply for the trace and logdet terms.
        """
        KL = 0.5*tf.reduce_sum(tf.square(self.q_mu))  # Mahalanobis term
        KL += -0.5*tf.cast(tf.size(self.q_mu), tf.float64)  # Constant term.
        if self.q_diag:
            logdets = [tf.reduce_sum(tf.log(tf.square(q_sqrt_d))) for q_sqrt_d in self.q_sqrt_kron]
            q_diag = tf.transpose(make_kvs([tf.reshape(tf.square(q_sqrt_d), (1, -1)) for q_sqrt_d in self.q_sqrt_kron]))
            # AiW = self.q_sqrt_lowrank / q_diag
            # IAWA = eye(2) + tf.matmul(tf.transpose(self.q_sqrt_lowrank), AiW)
            KL += 0.5 * tf.reduce_sum(q_diag)  # kron-trace
        else:
            Ls = [tf.batch_matrix_band_part(q_sqrt_d, -1, 0) for q_sqrt_d in self.q_sqrt_kron]
            logdets = [tf.reduce_sum(tf.log(tf.square(tf.diag_part(L)))) for L in Ls]
            # LiW = kron_mat_triangular_solve(Ls, self.q_sqrt_lowrank, num_cols=2, lower=True)
            # IAWA = eye(2) + tf.matmul(tf.transpose(LiW), LiW)
            KL += 0.5 * reduce(tf.mul, [tf.reduce_sum(tf.square(L)) for L in Ls])  # kron-trace
        N_others = [tf.cast(tf.size(self.q_mu) / tf.shape(q_sqrt_d)[0], tf.float64) for q_sqrt_d in self.q_sqrt_kron]
        KL += -0.5 * reduce(tf.add, [N*logdet for N, logdet in zip(N_others, logdets)])  # kron-logdet
        # KL += -tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(IAWA))))  # low-rank-log-det
        # KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt_lowrank))  # low-rank-trace
        return KL

    def build_likelihood(self):
        # compute the mean and variance of the latent function
        f_mu, f_var = self.build_predict(self.X, full_cov=False)

        E_lik = self.likelihood.variational_expectations(f_mu, f_var, self.Y)
        return tf.reduce_sum(E_lik) - self.build_KL()
