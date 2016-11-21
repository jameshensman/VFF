from __future__ import print_function, absolute_import
import numpy as np
import GPflow
import tensorflow as tf
from GPflow.tf_wraps import eye
from sfgp.spectral_covariance import make_Kuu, make_Kuf
from GPflow import settings
float_type = settings.dtypes.float_type


class GPR_1d(GPflow.model.GPModel):
    def __init__(self, X, Y, ms, a, b, kern,
                 mean_function=GPflow.mean_functions.Zero()):
        """
        In this special edition of VFF-GPR, we allow the boundary to be inside the data.

        This version is not very efficient. If recomputes the Kuf matrix at
        each iteration, and does not precompute any quantites, and does not
        exploit Kuu's strcture.

        Designed only for a demonstration with a, b, inside the data limits,
        for a practical version, use the VFF package.
        """
        assert X.shape[1] == 1
        assert isinstance(kern, (GPflow.kernels.Matern12,
                                 GPflow.kernels.Matern32,
                                 GPflow.kernels.Matern52))
        kern = kern
        likelihood = GPflow.likelihoods.Gaussian()
        GPflow.model.GPModel.__init__(self, X, Y, kern,
                                      likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

    def build_likelihood(self):
        num_inducing = tf.size(self.ms)
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = make_Kuf(self.kern, self.X, self.a, self.b, self.ms)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        Kuu = Kuu.get()
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf) / sigma
        AAT = tf.matmul(A, tf.transpose(A))

        B = AAT + eye(num_inducing * 2 - 1)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, err)) / sigma

        # compute log marginal bound
        ND = tf.cast(num_data * output_dim, float_type)
        D = tf.cast(output_dim, float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * self.likelihood.variance)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(err))/self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * tf.reduce_sum(Kdiag)/self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.diag_part(AAT))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        num_inducing = tf.size(self.ms)

        err = self.Y - self.mean_function(self.X)
        Kuf = make_Kuf(self.kern, self.X, self.a, self.b, self.ms)
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        Kuu = Kuu.get()
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf) / sigma
        AAT = tf.matmul(A, tf.transpose(A))

        B = AAT + eye(num_inducing * 2 - 1)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, err)) / sigma

        Kus = make_Kuf(self.kern, Xnew, self.a, self.b, self.ms)
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = self.kern.K(Xnew) + \
                tf.matmul(tf.transpose(tmp2), tmp2) - \
                tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + \
                tf.reduce_sum(tf.square(tmp2), 0) - \
                tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.pack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var
