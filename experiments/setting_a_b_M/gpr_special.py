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
import numpy as np
import gpflow
import tensorflow as tf
from vff.spectral_covariance import make_Kuu, make_Kuf
from gpflow import settings

float_type = settings.dtypes.float_type


class GPR_1d(gpflow.models.GPModel):
    def __init__(self, X, Y, ms, a, b, kern, mean_function=gpflow.mean_functions.Zero()):
        """
        In this special edition of VFF-GPR, we allow the boundary to be inside the data.

        This version is not very efficient. If recomputes the Kuf matrix at
        each iteration, and does not precompute any quantites, and does not
        exploit Kuu's strcture.

        Designed only for a demonstration with a, b, inside the data limits,
        for a practical version, use the VFF package.
        """
        assert X.shape[1] == 1
        assert isinstance(
            kern, (gpflow.kernels.Matern12, gpflow.kernels.Matern32, gpflow.kernels.Matern52)
        )
        kern = kern
        likelihood = gpflow.likelihoods.Gaussian()
        gpflow.models.GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

    @gpflow.params_as_tensors
    def _build_likelihood(self):
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

        B = AAT + tf.eye(num_inducing * 2 - 1, dtype=float_type)
        LB = tf.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, err)) / sigma

        # compute log marginal bound
        ND = tf.cast(num_data * output_dim, float_type)
        D = tf.cast(output_dim, float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * self.likelihood.variance)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.diag_part(AAT))

        return bound

    @gpflow.params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
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

        B = AAT + tf.eye(num_inducing * 2 - 1, dtype=float_type)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, err)) / sigma

        Kus = make_Kuf(self.kern, Xnew, self.a, self.b, self.ms)
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = (
                self.kern.K(Xnew)
                + tf.matmul(tf.transpose(tmp2), tmp2)
                - tf.matmul(tf.transpose(tmp1), tmp1)
            )
            var = var[:, :, None] * tf.ones(self.Y.shape[1], dtype=float_type)
        else:
            var = (
                self.kern.Kdiag(Xnew)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = var[:, None]  #  * tf.ones(self.Y.shape[1], dtype=float_type)
        return mean + self.mean_function(Xnew), var
