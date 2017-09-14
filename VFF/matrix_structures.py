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


import tensorflow as tf
from GPflow import settings
from functools import reduce
float_type = settings.dtypes.float_type
import numpy as np


class BlockDiagMat_many:
    def __init__(self, mats):
        self.mats = mats

    @property
    def shape(self):
        return (sum([m.shape[0] for m in mats]), sum([m.shape[1] for m in mats]))

    @property
    def sqrt_dims(self):
        return sum([m.sqrt_dims for m in mats])

    def _get_rhs_slices(self, X):
        ret = []
        start = 0
        for m in self.mats:
            ret.append(tf.slice(X, begin=tf.stack([start, 0]), size=tf.stack([m.shape[1], -1])))
            start = start + m.shape[1]
        return ret

    def _get_rhs_blocks(self, X):
        """
        X is a solid matrix, same size as this one. Get the blocks of X that
        correspond to the structure of this matrix
        """
        ret = []
        start1 = 0
        start2 = 0
        for m in self.mats:
            ret.append(tf.slice(X, begin=tf.stack([start1, start2]), size=m.shape))
            start1 = start1 + m.shape[0]
            start2 = start2 + m.shape[1]
        return ret

    def get(self):
        ret = self.mats[0].get()
        for m in self.mats[1:]:
            tr_shape = tf.stack([tf.shape(ret)[0], m.shape[1]])
            bl_shape = tf.stack([m.shape[0], tf.shape(ret)[1]])
            top = tf.concat([ret, tf.zeros(tr_shape, float_type)], axis=1)
            bottom = tf.concat([tf.zeros(bl_shape, float_type), m.get()], axis=1)
            ret = tf.concat([top, bottom], axis=0)
        return ret

    def logdet(self):
        return reduce(tf.add, [m.logdet() for m in self.mats])

    def matmul(self, X):
        return tf.concat([m.matmul(Xi) for m, Xi in zip(self.mats, self._get_rhs_slices(X))], axis=0)

    def solve(self, X):
        return tf.concat([m.solve(Xi) for m, Xi in zip(self.mats, self._get_rhs_slices(X))], axis=0)

    def inv(self):
        return BlockDiagMat_many([mat.inv() for mat in self.mats])

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        return reduce(tf.add, [m.trace_KiX(Xi) for m, Xi in zip(self.mats, self._get_rhs_blocks(X))])

    def get_diag(self):
        return tf.concat([m.get_diag() for m in self.mats], axis=0)

    def inv_diag(self):
        return tf.concat([m.inv_diag() for m in self.mats], axis=0)

    def matmul_sqrt(self, X):
        return tf.concat([m.matmul_sqrt(Xi) for m, Xi in zip(self.mats, self._get_rhs_slices(X))], axis=0)

    def matmul_sqrt_transpose(self, X):
        ret = []
        start = np.zeros((2, np.int32))
        for m in self.mats:
            ret.append(m.matmul_sqrt_transpose(tf.slice(X, begin=start, size=tf.stack([m.sqrt_dims, -1]))))
            start[0] += m.sqrt_dims

        return tf.concat(ret, axis=0)


class BlockDiagMat:
    def __init__(self, A, B):
        self.A, self.B = A, B

    @property
    def shape(self):
        mats = [self.A, self.B]
        return (sum([m.shape[0] for m in mats]), sum([m.shape[1] for m in mats]))

    @property
    def sqrt_dims(self):
        mats = [self.A, self.B]
        return sum([m.sqrt_dims for m in mats])

    def _get_rhs_slices(self, X):
        # X1 = X[:self.A.shape[1], :]
        X1 = tf.slice(X, begin=tf.zeros((2,), tf.int32), size=tf.stack([self.A.shape[1], -1]))
        # X2 = X[self.A.shape[1]:, :]
        X2 = tf.slice(X, begin=tf.stack([self.A.shape[1], 0]), size=-tf.ones((2,), tf.int32))
        return X1, X2

    def get(self):
        tl_shape = tf.stack([self.A.shape[0], self.B.shape[1]])
        br_shape = tf.stack([self.B.shape[0], self.A.shape[1]])
        top = tf.concat([self.A.get(), tf.zeros(tl_shape, float_type)], axis=1)
        bottom = tf.concat([tf.zeros(br_shape, float_type), self.B.get()], axis=1)
        return tf.concat([top, bottom], axis=0)

    def logdet(self):
        return self.A.logdet() + self.B.logdet()

    def matmul(self, X):
        X1, X2 = self._get_rhs_slices(X)
        top = self.A.matmul(X1)
        bottom = self.B.matmul(X2)
        return tf.concat([top, bottom], axis=0)

    def solve(self, X):
        X1, X2 = self._get_rhs_slices(X)
        top = self.A.solve(X1)
        bottom = self.B.solve(X2)
        return tf.concat([top, bottom], axis=0)

    def inv(self):
        return BlockDiagMat(self.A.inv(), self.B.inv())

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        X1, X2 = tf.slice(X, [0, 0], self.A.shape), tf.slice(X, self.A.shape, [-1, -1])
        top = self.A.trace_KiX(X1)
        bottom = self.B.trace_KiX(X2)
        return top + bottom

    def get_diag(self):
        return tf.concat([self.A.get_diag(), self.B.get_diag()], axis=0)

    def inv_diag(self):
        return tf.concat([self.A.inv_diag(), self.B.inv_diag()], axis=0)

    def matmul_sqrt(self, X):
        X1, X2 = self._get_rhs_slices(X)
        top = self.A.matmul_sqrt(X1)
        bottom = self.B.matmul_sqrt(X2)
        return tf.concat([top, bottom], axis=0)

    def matmul_sqrt_transpose(self, X):
        X1 = tf.slice(X, begin=tf.zeros((2,), tf.int32), size=tf.stack([self.A.sqrt_dims, -1]))
        X2 = tf.slice(X, begin=tf.stack([self.A.sqrt_dims, 0]), size=-tf.ones((2,), tf.int32))
        top = self.A.matmul_sqrt_transpose(X1)
        bottom = self.B.matmul_sqrt_transpose(X2)

        return tf.concat([top, bottom], axis=0)


class LowRankMat:
    def __init__(self, d, W):
        """
        A matrix of the form

            diag(d) + W W^T

        """
        self.d = d
        self.W = W

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    @property
    def sqrt_dims(self):
        return tf.size(self.d) + tf.shape(W)[1]

    def get(self):
        return tf.diag(self.d) + tf.matmul(self.W, tf.transpose(self.W))

    def logdet(self):
        part1 = tf.reduce_sum(tf.log(self.d))
        I = tf.eye(tf.shape(self.W)[1], float_type)
        M = I + tf.matmul(tf.transpose(self.W) / self.d, self.W)
        part2 = 2*tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(M))))
        return part1 + part2

    def matmul(self, B):
        WTB = tf.matmul(tf.transpose(self.W), B)
        WWTB = tf.matmul(self.W, WTB)
        DB = tf.reshape(self.d, [-1, 1]) * B
        return DB + WWTB

    def get_diag(self):
        return self.d + tf.reduce_sum(tf.square(self.W), 1)

    def solve(self, B):
        d_col = tf.expand_dims(self.d, 1)
        DiB = B / d_col
        DiW = self.W / d_col
        WTDiB = tf.matmul(tf.transpose(DiW), B)
        M = tf.eye(tf.shape(self.W)[1], float_type) + tf.matmul(tf.transpose(DiW), self.W)
        L = tf.cholesky(M)
        tmp1 = tf.matrix_triangular_solve(L, WTDiB, lower=True)
        tmp2 = tf.matrix_triangular_solve(tf.transpose(L), tmp1, lower=False)
        return DiB - tf.matmul(DiW, tmp2)

    def inv(self):
        di = tf.reciprocal(self.d)
        d_col = tf.expand_dims(self.d, 1)
        DiW = self.W / d_col
        M = tf.eye(tf.shape(self.W)[1], float_type) + tf.matmul(tf.transpose(DiW), self.W)
        L = tf.cholesky(M)
        v = tf.transpose(tf.matrix_triangular_solve(L, tf.transpose(DiW), lower=True))
        return LowRankMatNeg(di, V)

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        d_col = tf.expand_dims(self.d, 1)
        R = self.W / d_col
        RTX = tf.matmul(tf.transpose(R), X)
        RTXR = tf.matmul(RTX, R)
        M = tf.eye(tf.shape(self.W)[1], float_type) + tf.matmul(tf.transpose(R), self.W)
        Mi = tf.matrix_inverse(M)
        return tf.reduce_sum(tf.diag_part(X) * 1./self.d) - tf.reduce_sum(RTXR * Mi)

    def inv_diag(self):
        d_col = tf.expand_dims(self.d, 1)
        WTDi = tf.transpose(self.W / d_col)
        M = tf.eye(tf.shape(self.W)[1], float_type) + tf.matmul(WTDi, self.W)
        L = tf.cholesky(M)
        tmp1 = tf.matrix_triangular_solve(L, WTDi, lower=True)
        return 1./self.d - tf.reduce_sum(tf.square(tmp1), 0)

    def matmul_sqrt(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   W^T  ]

        This method right-multiplies the sqrt by the matrix B
        """

        DB = tf.expand_dims(tf.sqrt(self.d), 1) * B
        VTB = tf.matmul(tf.transpose(self.W), B)
        return tf.concat([DB, VTB], axis=0)

    def matmul_sqrt_transpose(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   W^T  ]

        This method right-multiplies the transposed-sqrt by the matrix B
        """
        B1 = tf.slice(B, tf.zeros((2,), tf.int32), tf.stack([tf.size(self.d), -1]))
        B2 = tf.slice(B, tf.stack([tf.size(self.d), 0]), -tf.ones((2,), tf.int32))
        return tf.expand_dims(tf.sqrt(self.d), 1) * B1 + tf.matmul(self.W, B2)


class LowRankMatNeg:
    def __init__(self, d, W):
        """
        A matrix of the form

            diag(d) - W W^T

        (note the minus sign)
        """
        self.d = d
        self.W = W

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    def get(self):
        return tf.diag(self.d) - tf.matmul(self.W, tf.transpose(self.W))




class Rank1Mat:
    def __init__(self, d, v):
        """
        A matrix of the form

            diag(d) + v v^T

        """
        self.d = d
        self.v = v

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    @property
    def sqrt_dims(self):
        return tf.size(self.d) + 1

    def get(self):
        V = tf.expand_dims(self.v, 1)
        return tf.diag(self.d) + tf.matmul(V, tf.transpose(V))

    def logdet(self):
        return tf.reduce_sum(tf.log(self.d)) +\
            tf.log(1. + tf.reduce_sum(tf.square(self.v) / self.d))

    def matmul(self, B):
        V = tf.expand_dims(self.v, 1)
        return tf.expand_dims(self.d, 1) * B +\
            tf.matmul(V, tf.matmul(tf.transpose(V), B))

    def solve(self, B):
        div = self.v / self.d
        c = 1. + tf.reduce_sum(div * self.v)
        div = tf.expand_dims(div, 1)
        return B / tf.expand_dims(self.d, 1) -\
            tf.matmul(div/c, tf.matmul(tf.transpose(div), B))

    def inv(self):
        di = tf.reciprocal(self.d)
        Div = self.v * di
        M = 1. + tf.reduce_sum(Div * self.v)
        v_new = Div / tf.sqrt(M)
        return Rank1MatNeg(di, v_new)

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        R = tf.expand_dims(self.v / self.d, 1)
        RTX = tf.matmul(tf.transpose(R), X)
        RTXR = tf.matmul(RTX, R)
        M = 1 + tf.reduce_sum(tf.square(self.v) / self.d)
        return tf.reduce_sum(tf.diag_part(X) / self.d) - RTXR / M

    def get_diag(self):
        return self.d + tf.square(self.v)

    def inv_diag(self):
        div = self.v / self.d
        c = 1. + tf.reduce_sum(div * self.v)
        return 1./self.d - tf.square(div) / c

    def matmul_sqrt(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   V^T  ]

        This method right-multiplies the sqrt by the matrix B
        """

        DB = tf.expand_dims(tf.sqrt(self.d), 1) * B
        VTB = tf.matmul(tf.expand_dims(self.v, 0), B)
        return tf.concat([DB, VTB], axis=0)

    def matmul_sqrt_transpose(self, B):
        """
        There's a non-square sqrt of this matrix given by
          [ D^{1/2}]
          [   W^T  ]

        This method right-multiplies the transposed-sqrt by the matrix B
        """
        B1 = tf.slice(B, tf.zeros((2,), tf.int32), tf.stack([tf.size(self.d), -1]))
        B2 = tf.slice(B, tf.stack([tf.size(self.d), 0]), -tf.ones((2,), tf.int32))
        return tf.expand_dims(tf.sqrt(self.d), 1) * B1 + tf.matmul(tf.expand_dims(self.v, 1), B2)


class Rank1MatNeg:
    def __init__(self, d, v):
        """
        A matrix of the form

            diag(d) - v v^T

        (note the minus sign)
        """
        self.d = d
        self.v = v

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    def get(self):
        W = tf.expand_dims(self.v, 1)
        return tf.diag(self.d) - tf.matmul(W, tf.transpose(W))


class DiagMat:
    def __init__(self, d):
        self.d = d

    @property
    def shape(self):
        return (tf.size(self.d), tf.size(self.d))

    @property
    def sqrt_dims(self):
        return tf.size(self.d)

    def get(self):
        return tf.diag(self.d)

    def logdet(self):
        return tf.reduce_sum(tf.log(self.d))

    def matmul(self, B):
        return tf.expand_dims(self.d, 1) * B

    def solve(self, B):
        return B / tf.expand_dims(self.d, 1)

    def inv(self):
        return DiagMat(tf.reciprocal(self.d))

    def trace_KiX(self, X):
        """
        X is a square matrix of the same size as this one.
        if self is K, compute tr(K^{-1} X)
        """
        return tf.reduce_sum(tf.diag_part(X) / self.d)

    def get_diag(self):
        return self.d

    def inv_diag(self):
        return 1. / self.d

    def matmul_sqrt(self, B):
        return tf.expand_dims(tf.sqrt(self.d), 1) * B

    def matmul_sqrt_transpose(self, B):
        return tf.expand_dims(tf.sqrt(self.d), 1) * B
