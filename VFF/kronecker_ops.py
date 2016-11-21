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
from functools import reduce
import GPflow
import numpy as np


def kron_two(A, B):
    """compute the Kronecker product of two tensorfow tensors"""
    shape = tf.pack([tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]])
    return tf.reshape(tf.expand_dims(tf.expand_dims(A, 1), 3) * tf.expand_dims(tf.expand_dims(B, 0), 2), shape)


def kron(K):
    return reduce(kron_two, K)


def kron_vec_mul(K, vec):
    """
    K is a list of tf_arrays to be kroneckered
    vec is a N x 1 tf_array
    """
    N_by_1 = tf.pack([tf.size(vec), 1])

    def f(v, k):
        v = tf.reshape(v, tf.pack([tf.shape(k)[1], -1]))
        v = tf.matmul(k, v)
        return tf.reshape(tf.transpose(v), N_by_1)  # transposing first flattens the vector in column order
    return reduce(f, K, vec)


def kron_mat_mul(K, mat, num_cols):
    return tf.concat(1, [kron_vec_mul(K, mat[:, i:i+1]) for i in range(num_cols)])


def kron_vec_triangular_solve(L, vec, lower=True):
    """
    L is a list of lower-triangular tf_arrays to be kroneckered
    vec is a N x 1 tf_array
    """
    N_by_1 = tf.pack([tf.size(vec), 1])

    def f(v, L_d):
        v = tf.reshape(v, tf.pack([tf.shape(L_d)[1], -1]))
        v = tf.matrix_triangular_solve(L_d, v, lower=lower)
        return tf.reshape(tf.transpose(v), N_by_1)  # transposing first flattens the vector in column order
    return reduce(f, L, vec)


def kron_mat_triangular_solve(L, mat, num_cols, lower=True):
    return tf.concat(1, [kron_vec_triangular_solve(L, mat[:, i:i+1], lower=lower) for i in range(num_cols)])


def make_kvs_two(A, B):
    """
    compute the Kronecer-Vector stack of the matrices A and B
    """
    shape = tf.pack([tf.shape(A)[0], tf.shape(A)[1] * tf.shape(B)[1]])
    return tf.reshape(tf.batch_matmul(tf.expand_dims(A, 2), tf.expand_dims(B, 1)), shape)


def make_kvs(k):
    """Compute the kronecker-vector stack of the list of matrices k"""
    return reduce(make_kvs_two, k)


def kron_vec_apply(K, vec, attr):
    """
    K is a list of objects to be kroneckered
    vec is a N x 1 tf_array
    attr is a string rep

    each element of k must have a method corresponding to the name 'attr' (e.g. matmul, solve...)
    """
    N_by_1 = tf.pack([-1, 1])

    def f(v, k):
        v = tf.reshape(v, tf.pack([k.shape[1], -1]))
        v = getattr(k, attr)(v)
        return tf.reshape(tf.transpose(v), N_by_1)  # transposing first flattens the vector in column order
    return reduce(f, K, vec)


def kron_mat_apply(K, mat, attr, num_cols):
    return tf.concat(1, [kron_vec_apply(K, mat[:, i:i+1], attr) for i in range(num_cols)])


def kvs_dot_vec_memhungry(k, c):
    """
    kvs is a kron-vec stack. A matrix where each row is a kronecker product of
    row-vectors.  This implementation requires a lot of memory!
    """
    N = tf.shape(k[0])[0]
    # we need to repeat the vec because batch_matmul won't broadcast.
    C = tf.tile(tf.reshape(c, (1, -1, 1)), tf.pack([N, 1, 1]))  # C is N x D_total x 1
    for ki in k:
        Di = tf.shape(ki)[1]
        # this transpose works around reshaping in fortran order...
        C = tf.transpose(tf.reshape(C, tf.pack([N, Di, -1])), perm=[0, 2, 1])
        # C is now N x (D5 D4 D3 ...) x Di
        C = tf.batch_matmul(C, tf.expand_dims(ki, 2))
        # C is now N x (D5 D4 D3...) x 1
    return tf.reshape(C, tf.pack([N, 1]))  # squeeze out the extra dim


def kvs_dot_vec_loop(k, c):
    """
    kvs is a kron-vec stack. A matrix where each row is a kronecker product of
    row-vectors.

    here we loop over the rows of the k-matrices
    """

    def inner(k_list, c_vec):
        """
        k_list is a list of 1 x Di matrices
        """
        for ki in k_list:
            Di = tf.shape(ki)[1]
            c_vec = tf.transpose(tf.reshape(c_vec, tf.pack([Di, -1])))
            c_vec = tf.matmul(c_vec, tf.transpose(ki))
        return c_vec

    N = tf.shape(k[0])[0]
    i = tf.constant(0)
    ret = tf.zeros(0, GPflow.settings.dtypes.float_type)

    def body(i, ret):
        ret_i = inner([tf.slice(kd, [i, 0], [1, -1]) for kd in k], c)
        return i + 1, tf.concat(0, [ret, tf.reshape(ret_i, [1])])

    def cond(i, ret):
        return tf.less(i, N)

    _, ret = tf.while_loop(cond, body, [i, ret])
    return tf.reshape(ret, tf.pack([N, 1]))


def kvs_dot_vec_specialfirst(k, c):
    """
    kvs is a kron-vec stack. A matrix where each row is a kronecker product of
    row-vectors.  This implementation requires less memory than the memhungry
    version (I hope), and is much faster than the looped version.
    """
    N = tf.shape(k[0])[0]
    # do the first matmul in a special way that saves us from tiling...
    D0 = tf.shape(k[0])[1]
    C = tf.transpose(tf.reshape(c, tf.pack([D0, -1])))
    C = tf.transpose(tf.matmul(C, tf.transpose(k[0])))
    C = tf.expand_dims(C, 2)

    # do the remaining matmuls in the same way as the memhungry version
    for ki in k[1:]:
        Di = tf.shape(ki)[1]
        # this transpose works around reshaping in fortran order...
        C = tf.transpose(tf.reshape(C, tf.pack([N, Di, -1])), perm=[0, 2, 1])
        # C is now N x (D5 D4 D3 ...) x Di
        C = tf.batch_matmul(C, tf.expand_dims(ki, 2))
        # C is now N x (D5 D4 D3...) x 1
    return tf.reshape(C, tf.pack([N, 1]))  # squeeze out the extra dim


def kvs_dot_vec(k, c):
    return kvs_dot_vec_specialfirst(k, c)


def kvs_dot_mat(K, mat, num_cols):
    return tf.concat(1, [kvs_dot_vec(K, mat[:, i:i+1]) for i in range(num_cols)])


def log_det_kron_sum_np(L1, L2):
    """
    """
    L1_logdets = [np.sum(np.log(np.square(np.diag(L)))) for L in L1]
    total_size = np.prod([L.shape[0] for L in L1])
    N_other = [total_size / L.shape[0] for L in L1]
    L1_logdet = np.sum([s*ld for s, ld in zip(N_other, L1_logdets)])
    LiL = [np.linalg.solve(L, R) for L, R in zip(L1, L2)]
    eigvals = [np.linalg.eigvals(np.dot(mat, mat.T)) for mat in LiL]
    return np.sum(np.log(1 + reduce(np.kron, eigvals))) + L1_logdet


def log_det_kron_sum(L1, L2):
    """
    L1 is a list of lower triangular arrays.
    L2 is a list of lower triangular arrays.

    if S1 = kron(L1) * kron(L1).T, and S2 similarly,
    this function computes the log determinant of S1 + S2
    """
    L1_logdets = [tf.reduce_sum(tf.log(tf.square(tf.diag_part(L)))) for L in L1]
    total_size = reduce(tf.mul, [L.shape[0] for L in L1])
    N_other = [total_size / tf.shape(L)[0] for L in L1]
    L1_logdet = reduce(tf.add, [s*ld for s, ld in zip(N_other, L1_logdets)])
    LiL = [tf.matrix_triangular_solve(L, R) for L, R in zip(L1, L2)]
    eigvals = [tf.self_adjoint_eigvals(tf.matmul(mat, mat.T)) for mat in LiL]
    eigvals_kronned = kron_vec_mul([tf.reshape(e, [-1, 1]) for e in eigvals], tf.ones([1, 1], tf.float64))
    return tf.reduce_sum(tf.log(1 + eigvals_kronned)) + L1_logdet


if __name__ == '__main__':

    K = [np.random.randn(1000, 400) for i in range(2)]
    c = np.random.randn(400**2, 1)

    sess = tf.Session()
    r1 = kvs_dot_vec_memhungry(K, c)
    r2 = kvs_dot_vec_loop(K, c)
    r3 = kvs_dot_vec_specialfirst(K, c)
    rr1 = sess.run(r1)
    rr2 = sess.run(r2)
    rr3 = sess.run(r3)
