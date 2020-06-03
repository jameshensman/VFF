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


from functools import reduce

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.utilities import to_default_float


from .matrix_structures import BlockDiagMat_many
from .spectral_covariance import make_Kuf, make_Kuf_np, make_Kuu


class GPR_additive(gpflow.models.GPR):

    def __init__(self, data, ms, a, b, kernels):
        X, Y = data
        assert X.shape[1] == len(kernels)
        assert a.size == len(kernels)
        assert b.size == len(kernels)
        # Check that the type of each kernel belongs to the Matern family
        assert all(map(lambda k: "Matern" in type(k).__name__, kernels))

        mean_function = gpflow.mean_functions.Zero()
        super().__init__(data, kernels, mean_function)

        self.X, self.Y = data
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.a = a
        self.b = b
        self.ms = ms

        # pre compute static quantities: chunk data to save memory
        Mtotal = (2*self.ms.size - 1) * X.shape[1]
        self.KufY = np.zeros((Mtotal, 1))
        self.KufKfu = np.zeros((Mtotal, Mtotal))
        for i in range(0, (X.shape[0]), 10000):
            Xchunk = X[i:i + 10000]
            Ychunk = Y[i:i + 10000]
            Kuf_chunk = np.empty((0, Xchunk.shape[0]))
            KufY_chunk = np.empty((0, Ychunk.shape[1]))
            for i, (ai, bi) in enumerate(zip(self.a, self.b)):
                assert np.all(Xchunk[:, i] > ai)
                assert np.all(Xchunk[:, i] < bi)
                Kuf = make_Kuf_np(Xchunk[:, i:i+1], ai, bi, self.ms)
                KufY_chunk = np.vstack((KufY_chunk, np.dot(Kuf, Ychunk)))
                Kuf_chunk = np.vstack((Kuf_chunk, Kuf))
            self.KufKfu += np.dot(Kuf_chunk, Kuf_chunk.T)
            self.KufY += KufY_chunk

        self.tr_YTY = tf.convert_to_tensor(np.sum(np.square(Y)))
        self.KufY = tf.convert_to_tensor(self.KufY)
        self.KufKfu = tf.convert_to_tensor(self.KufKfu)
    
    def log_marginal_likelihood(self):
        X, Y = self.data
        num_data = tf.shape(Y)[0]
        output_dim = tf.shape(Y)[1]

        total_variance = reduce(tf.add, [k.variance for k in self.kernel])
        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kernel, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.linalg.cholesky(P)
        log_det_P = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L))))
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        # compute log marginal bound
        ND = to_default_float(num_data * output_dim)
        D = to_default_float(output_dim)
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_P
        bound += 0.5 * D * Kuu.logdet()
        bound += -0.5 * self.tr_YTY / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * ND * total_variance / sigma2
        bound += 0.5 * D * Kuu.trace_KiX(self.KufKfu) / sigma2

        return bound

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        assert not full_output_cov

        Kuu = [make_Kuu(k, ai, bi, self.ms) for k, ai, bi in zip(self.kernel, self.a, self.b)]
        Kuu = BlockDiagMat_many([mat for k in Kuu for mat in [k.A, k.B]])
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        P = self.KufKfu / sigma2 + Kuu.get()
        L = tf.linalg.cholesky(P)
        c = tf.linalg.triangular_solve(L, self.KufY) / sigma2

        values = [
            make_Kuf(k, Xnew[:, i:i+1], a, b, self.ms)
            for i, (k, a, b) in enumerate(zip(self.kernel, self.a, self.b))
        ]
        Kus = tf.concat(values, axis=0)
        tmp = tf.linalg.triangular_solve(L, Kus)
        # bla
        mean = tf.matmul(tmp, c, transpose_a=True)
        KiKus = Kuu.solve(Kus)
        if full_cov:
            var = reduce(tf.add, [k.K(Xnew[:, i:i+1]) for i, k in enumerate(self.kernel)])
            var += tf.matmul(tf.transpose(tmp), tmp)
            var -= tf.matmul(tf.transpose(KiKus), Kus)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = reduce(tf.add, [k.K_diag(Xnew[:, i:i+1]) for i, k in enumerate(self.kernel)])
            var += tf.reduce_sum(tf.square(tmp), 0)
            var -= tf.reduce_sum(KiKus * Kus, 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean, var
