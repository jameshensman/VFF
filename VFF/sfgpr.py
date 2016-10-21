from __future__ import print_function, absolute_import
import numpy as np
import GPflow
import tensorflow as tf
from functools import reduce
from .spectral_covariance import make_Kuu, make_Kuf, make_Kuf_np
from .matrix_structures import BlockDiagMat_many
from GPflow import settings
float_type = settings.dtypes.float_type


class SFGPR_1d(GPflow.model.GPModel):
    def __init__(self, X, Y, ms, a, b, kern):
        assert X.shape[1] == 1
        assert isinstance(kern, (GPflow.kernels.Matern12,
                                 GPflow.kernels.Matern32,
                                 GPflow.kernels.Matern52))
        likelihood = GPflow.likelihoods.Gaussian()
        mean_function = GPflow.mean_functions.Zero()
        GPflow.model.GPModel.__init__(self, X, Y, kern,
                                      likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

        # pre compute static quantities
        assert np.all(X > a)
        assert np.all(X < b)
        Kuf = make_Kuf_np(X, a, b, ms)
        self.KufY = np.dot(Kuf, Y)
        self.KufKfu = np.dot(Kuf, Kuf.T)
        self.tr_YTY = np.sum(np.square(Y))

    def build_likelihood(self):
        Kdiag = self.kern.Kdiag(self.X)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.cholesky(P)
        log_det_P = tf.reduce_sum(tf.log(tf.square(tf.diag_part(L))))
        c = tf.matrix_triangular_solve(L, self.KufY) / sigma2

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), float_type)
        D = tf.cast(tf.shape(self.Y)[1], float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_P
        bound += 0.5 * D * Kuu.logdet()
        bound += -0.5 * self.tr_YTY / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * tf.reduce_sum(Kdiag)/sigma2
        bound += 0.5 * Kuu.trace_KiX(self.KufKfu) / sigma2

        return bound

    def build_predict(self, Xnew, full_cov=False):
        Kdiag = self.kern.Kdiag(self.X)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.cholesky(P)
        log_det_P = tf.reduce_sum(tf.log(tf.square(tf.diag_part(L))))
        c = tf.matrix_triangular_solve(L, self.KufY) / sigma2


        Kus = make_Kuf(self.kern, Xnew, self.a, self.b, self.ms)
        tmp = tf.matrix_triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = Kuu.solve(Kus)
        if full_cov:
            var = self.kern.k(xnew) + \
               tf.matmul(tf.transpose(tmp), tmp) - \
               tf.matmul(tf.transpose(kikus), kus)
            shape = tf.pack([1, 1, tf.shape(self.y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew)
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.pack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var
