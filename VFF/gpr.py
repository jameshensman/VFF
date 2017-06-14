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
import GPflow
import tensorflow as tf
from functools import reduce
from .spectral_covariance import make_Kuu, make_Kuf, make_Kuf_np
from .matrix_structures import BlockDiagMat_many
from GPflow import settings
float_type = settings.dtypes.float_type


class GPR_1d(GPflow.model.GPModel):
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
        Kuu = make_Kuu(self.kern, self.a, self.b, self.ms)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.cholesky(P)
        c = tf.matrix_triangular_solve(L, self.KufY) / sigma2

        Kus = make_Kuf(self.kern, Xnew, self.a, self.b, self.ms)
        tmp = tf.matrix_triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = Kuu.solve(Kus)
        if full_cov:
            var = self.kern.k(Xnew) + \
               tf.matmul(tf.transpose(tmp), tmp) - \
               tf.matmul(tf.transpose(KiKus), Kus)
            shape = tf.stack([1, 1, tf.shape(self.y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew)
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var


class GPR_additive(GPflow.model.GPModel):
    def __init__(self, X, Y, ms, a, b, kern_list):
        assert X.shape[1] == len(kern_list)
        assert a.size == len(kern_list)
        assert b.size == len(kern_list)
        for kern in kern_list:
            assert isinstance(kern, (GPflow.kernels.Matern12,
                                     GPflow.kernels.Matern32,
                                     GPflow.kernels.Matern52))
        likelihood = GPflow.likelihoods.Gaussian()
        mean_function = GPflow.mean_functions.Zero()
        GPflow.model.GPModel.__init__(self, X, Y, None,
                                      likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

        self.kerns = GPflow.param.ParamList(kern_list)

        # pre compute static quantities: chunk data to save memory
        self.tr_YTY = GPflow.param.DataHolder(np.sum(np.square(Y)))
        Mtotal = (2*self.ms.size - 1) * X.shape[1]
        self.KufY = np.zeros((Mtotal, 1))
        self.KufKfu = np.zeros((Mtotal, Mtotal))
        for i in range(0, (X.shape[0]), 10000):
            Xchunk = X[i:i + 10000]
            Ychunk = Y[i:i + 10000]
            Kuf_chunk = np.empty((0, Xchunk.shape[0]))
            KufY_chunk = np.empty((0, Ychunk.shape[1]))
            for i, (k, ai, bi) in enumerate(zip(self.kerns, self.a, self.b)):
                assert np.all(Xchunk[:, i] > ai)
                assert np.all(Xchunk[:, i] < bi)
                Kuf = make_Kuf_np(Xchunk[:, i:i+1], ai, bi, self.ms)
                KufY_chunk = np.vstack((KufY_chunk, np.dot(Kuf, Ychunk)))
                Kuf_chunk = np.vstack((Kuf_chunk, Kuf))
            self.KufKfu += np.dot(Kuf_chunk, Kuf_chunk.T)
            self.KufY += KufY_chunk
        self.KufY = GPflow.param.DataHolder(self.KufY)
        self.KufKfu = GPflow.param.DataHolder(self.KufKfu)

    def build_likelihood(self):
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        total_variance = reduce(tf.add, [k.variance for k in self.kerns])
        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kerns, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.cholesky(P)
        log_det_P = tf.reduce_sum(tf.log(tf.square(tf.diag_part(L))))
        c = tf.matrix_triangular_solve(L, self.KufY) / sigma2

        # compute log marginal bound
        ND = tf.cast(num_data * output_dim, float_type)
        D = tf.cast(output_dim, float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_P
        bound += 0.5 * D * Kuu.logdet()
        bound += -0.5 * self.tr_YTY / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * ND * total_variance / sigma2
        bound += 0.5 * D * Kuu.trace_KiX(self.KufKfu) / sigma2

        return bound

    def build_predict(self, Xnew, full_cov=False):
        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kerns, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.cholesky(P)
        c = tf.matrix_triangular_solve(L, self.KufY) / sigma2

        Kus = tf.concat([make_Kuf(k, Xnew[:, i:i+1], a, b, self.ms)
                         for i, (k, a, b) in enumerate(zip(self.kerns, self.a, self.b))], axis=0)
        tmp = tf.matrix_triangular_solve(L, Kus)
        mean = tf.matmul(tf.transpose(tmp), c)
        KiKus = Kuu.solve(Kus)
        if full_cov:
            var = reduce(tf.add, [k.K(Xnew[:, i:i+1]) for i, k in enumerate(self.kerns)])
            var += tf.matmul(tf.transpose(tmp), tmp)
            var -= tf.matmul(tf.transpose(KiKus), Kus)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = reduce(tf.add, [k.Kdiag(Xnew[:, i:i+1]) for i, k in enumerate(self.kerns)])
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var

    @GPflow.param.AutoFlow((float_type, [None, 1]))
    def predict_components(self, Xnew):
        """
        Here, Xnew should be a Nnew x 1 array of points at which to test each function
        """
        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kerns, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.cholesky(P)
        c = tf.matrix_triangular_solve(L, self.KufY) / sigma2

        Kus_blocks = [make_Kuf(k, Xnew, a, b, self.ms)
                      for i, (k, a, b) in enumerate(zip(self.kerns, self.a, self.b))]
        Kus = []
        start = tf.constant(0, tf.int32)
        for i, b in enumerate(Kus_blocks):
            zeros_above = tf.zeros(tf.stack([start, tf.shape(b)[1]]), float_type)
            zeros_below = tf.zeros(tf.stack([tf.shape(L)[0] - start - tf.shape(b)[0], tf.shape(b)[1]]), float_type)
            Kus.append(tf.concat([zeros_above, b, zeros_below], axis=0))
            start = start + tf.shape(b)[0]

        tmp = [tf.matrix_triangular_solve(L, Kus_i) for Kus_i in Kus]
        mean = [tf.matmul(tf.transpose(tmp_i), c) for tmp_i in tmp]
        KiKus = [Kuu.solve(Kus_i) for Kus_i in Kus]
        var = [k.Kdiag(Xnew[:, i:i+1]) for i, k in enumerate(self.kerns)]
        var = [v + tf.reduce_sum(tf.square(tmp_i), 0) for v, tmp_i in zip(var, tmp)]
        var = [v - tf.reduce_sum(KiKus_i * Kus_i, 0) for v, KiKus_i, Kus_i in zip(var, KiKus, Kus)]
        var = [tf.expand_dims(v, 1) for v in var]
        return tf.concat(mean, axis=1), tf.concat(var, axis=1)
