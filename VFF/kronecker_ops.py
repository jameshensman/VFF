# Copyright 2020 ST John
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
from gpflow import default_float
import numpy as np


def kron_two(A, B):
    """compute the Kronecker product of two tensorfow tensors"""
    shape = tf.stack([tf.shape(A)[0] * tf.shape(B)[0], tf.shape(A)[1] * tf.shape(B)[1]])
    return tf.reshape(
        tf.expand_dims(tf.expand_dims(A, 1), 3) * tf.expand_dims(tf.expand_dims(B, 0), 2), shape
    )


def kron(K):
    return reduce(kron_two, K)


def kron_mat_mul(Ks, X, num_cols=None):
    """
    :param Ks: list of matrices that are kroneckered together, e.g. [A, B, C] for (A kron B kron C)
    :param X: the dense matrix with which to right-multiply the kronecker matrix
    :param num_cols: ignored (backwards-compatibility)

    If 'Ks' is [A, B, C], with dimensions A: (m x n), B: (p x q), C: (r x s), then
    the kroneckered matrix has shape (m*p*r, n*q*s).
    'X' has to have shape (s*q*n, k).

    returns: (A kron B kron C) @ X, which has shape (r*p*m, k)

    The result can be rewritten as C [B [A X^T]^T]^T with appropriate intermediate reshapes.
    """

    def mykronvec(f, A):
        """
        f: intermediate result of the kronecker-matrix multiplication
           (in the first iteration, this is just `mat`)
        A: the next kronecker component to use

        returns $A f^T$, with appropriate reshapings
        """
        m, n = tf.shape(A)  # A.shape
        n_times_o, k = tf.shape(f)  # f.shape
        o = n_times_o // n

        # f contains 'extra dimensions' both in terms of the separate columns of X
        # as well as in terms of the separate Kronecker components. We need the
        # 'active' dimension for the current component in the middle: then tensorflow
        # broadcasting deals with one of them, and the matrix multiplication also
        # "implicitly" broadcasts over the columns of the right-hand matrix which
        # deals with the other.

        # the transpose ensures column-wise rather than row-wise reshaping
        mat_f_T = tf.reshape(tf.transpose(a=f), (k, n, o))
        # Note that there is some magic involved in the transpose incidentally ensuring the right
        # ordering... but it works as proved by the tests, so it should be all good...

        # tensorflow only broadcasts over the first dimension in matrix
        # multiplication when the length is the same, hence we need to tile
        # explicitly:
        A = A[None, :, :] * tf.ones((k, 1, 1), dtype=tf.float64)
        # A now has shape (k, m, n)

        v = tf.matmul(A, mat_f_T)  # shape (k, m, o)
        # transpose needed for column-wise reshaping
        vec_v = tf.reshape(tf.transpose(a=v), (o * m, k))
        return vec_v

    return reduce(mykronvec, Ks, X)


def kron_vec_mul(K, vec):
    """
    K is a list of tf_arrays to be kroneckered
    vec is a N x 1 tf_array
    """
    return kron_mat_mul(K, vec)


def kron_vec_triangular_solve(L, vec, lower=True):
    """
    L is a list of lower-triangular tf_arrays to be kroneckered
    vec is a N x 1 tf_array
    """
    N_by_1 = tf.stack([tf.size(vec), 1])

    def f(v, L_d):
        v = tf.reshape(v, tf.stack([tf.shape(L_d)[1], -1]))
        v = tf.linalg.triangular_solve(L_d, v, lower=lower)
        return tf.reshape(
            tf.transpose(v), N_by_1
        )  # transposing first flattens the vector in column order

    return reduce(f, L, vec)


def kron_mat_triangular_solve(L, mat, num_cols, lower=True):
    return tf.concat(
        [kron_vec_triangular_solve(L, mat[:, i : i + 1], lower=lower) for i in range(num_cols)],
        axis=1,
    )


def make_kvs_two(A, B):
    """
    compute the Kronecer-Vector stack of the matrices A and B
    """
    shape = tf.stack([tf.shape(A)[0], tf.shape(A)[1] * tf.shape(B)[1]])
    return tf.reshape(tf.matmul(tf.expand_dims(A, 2), tf.expand_dims(B, 1)), shape)


def make_kvs(k):
    """Compute the kronecker-vector stack of the list of matrices k"""
    return reduce(make_kvs_two, k)


def make_kvs_two_np(A, B):
    # return np.tile(A, [B.shape[0], 1]) * np.repeat(B, A.shape[0], axis=0)
    return np.repeat(A, B.shape[0], axis=0) * np.tile(B, [A.shape[0], 1])


def make_kvs_np(A_list):
    return reduce(make_kvs_two_np, A_list)


def kron_vec_apply(K, vec, attr):
    """
    K is a list of objects to be kroneckered
    vec is a N x 1 tf_array
    attr is a string rep

    each element of k must have a method corresponding to the name 'attr' (e.g. matmul, solve...)
    """
    N_by_1 = tf.stack([-1, 1])

    def f(v, k):
        v = tf.reshape(v, tf.stack([k.shape[1], -1]))
        v = getattr(k, attr)(v)
        return tf.reshape(
            tf.transpose(v), N_by_1
        )  # transposing first flattens the vector in column order

    return reduce(f, K, vec)


def kron_mat_apply(K, mat, attr, num_cols):
    return tf.concat([kron_vec_apply(K, mat[:, i : i + 1], attr) for i in range(num_cols)], axis=1)


def kvs_dot_vec_memhungry(k, c):
    """
    kvs is a kron-vec stack. A matrix where each row is a kronecker product of
    row-vectors.  This implementation requires a lot of memory!
    """
    N = tf.shape(k[0])[0]
    # we need to repeat the vec because matmul won't broadcast.
    C = tf.tile(tf.reshape(c, (1, -1, 1)), tf.stack([N, 1, 1]))  # C is N x D_total x 1
    for ki in k:
        Di = tf.shape(ki)[1]
        # this transpose works around reshaping in fortran order...
        C = tf.transpose(tf.reshape(C, tf.stack([N, Di, -1])), perm=[0, 2, 1])
        # C is now N x (D5 D4 D3 ...) x Di
        C = tf.matmul(C, tf.expand_dims(ki, 2))
        # C is now N x (D5 D4 D3...) x 1
    return tf.reshape(C, tf.stack([N, 1]))  # squeeze out the extra dim


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
            c_vec = tf.transpose(tf.reshape(c_vec, tf.stack([Di, -1])))
            c_vec = tf.matmul(c_vec, tf.transpose(ki))  # XXX
        return c_vec

    N = tf.shape(k[0])[0]
    i = tf.constant(0)
    ret = tf.zeros(0, default_float())

    def body(i, ret):
        ret_i = inner([tf.slice(kd, [i, 0], [1, -1]) for kd in k], c)
        return i + 1, tf.concat([ret, tf.reshape(ret_i, [1])], axis=0)

    def cond(i, ret):
        return tf.less(i, N)

    _, ret = tf.while_loop(cond, body, [i, ret])
    return tf.reshape(ret, tf.stack([N, 1]))


def kvs_dot_vec_specialfirst(k, c):
    """
    kvs is a kron-vec stack. A matrix where each row is a kronecker product of
    row-vectors.  This implementation requires less memory than the memhungry
    version (I hope), and is much faster than the looped version.
    """
    N = tf.shape(k[0])[0]
    # do the first matmul in a special way that saves us from tiling...
    D0 = tf.shape(k[0])[1]
    C = tf.transpose(tf.reshape(c, tf.stack([D0, -1])))
    C = tf.transpose(tf.matmul(C, tf.transpose(k[0])))  # XXX
    C = tf.expand_dims(C, 2)

    # do the remaining matmuls in the same way as the memhungry version
    for ki in k[1:]:
        Di = tf.shape(ki)[1]
        # this transpose works around reshaping in fortran order...
        C = tf.transpose(tf.reshape(C, tf.stack([N, Di, -1])), perm=[0, 2, 1])
        # C is now N x (D5 D4 D3 ...) x Di
        C = tf.matmul(C, tf.expand_dims(ki, 2))
        # C is now N x (D5 D4 D3...) x 1
    return tf.reshape(C, tf.stack([N, 1]))  # squeeze out the extra dim


def kvs_dot_vec(k, c):
    return kvs_dot_vec_specialfirst(k, c)


def kvs_dot_mat(K, mat, num_cols):
    return tf.concat([kvs_dot_vec(K, mat[:, i : i + 1]) for i in range(num_cols)], axis=1)


def log_det_kron_sum_np(L1, L2):
    """"""
    L1_logdets = [np.sum(np.log(np.square(np.diag(L)))) for L in L1]
    total_size = np.prod([L.shape[0] for L in L1])
    N_other = [total_size / L.shape[0] for L in L1]
    L1_logdet = np.sum([s * ld for s, ld in zip(N_other, L1_logdets)])
    LiL = [np.linalg.solve(L, R) for L, R in zip(L1, L2)]
    eigvals = [np.linalg.eigvals(np.dot(mat, mat.T)) for mat in LiL]
    return np.sum(np.log(1 + reduce(np.kron, eigvals))) + L1_logdet


def workaround__self_adjoint_eigvals(mat):
    """
    The gradient of tensorflow.self_adjoint_eigvals() is currently broken,
    see https://github.com/tensorflow/tensorflow/issues/11821

    This works (but is less efficient).
    """
    eigvals, _ = tf.linalg.eigh(mat)
    return eigvals


def log_det_kron_sum(L1, L2):
    """
    L1 is a list of lower triangular arrays.
    L2 is a list of lower triangular arrays.

    if S1 = kron(L1) * kron(L1).T, and S2 similarly,
    this function computes the log determinant of S1 + S2
    """
    L1_logdets = [tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L)))) for L in L1]
    total_size = reduce(tf.multiply, [tf.shape(L)[0] for L in L1])
    N_other = [total_size / tf.shape(L)[0] for L in L1]
    L1_logdet = reduce(tf.add, [s * ld for s, ld in zip(N_other, L1_logdets)])
    LiL = [tf.linalg.triangular_solve(L, R) for L, R in zip(L1, L2)]
    eigvals = [
        workaround__self_adjoint_eigvals(tf.matmul(mat, mat, transpose_b=True)) for mat in LiL
    ]
    eigvals_kronned = kron_vec_mul(
        [tf.reshape(e, [-1, 1]) for e in eigvals], tf.ones([1, 1], tf.float64)
    )
    return tf.reduce_sum(tf.math.log(1 + eigvals_kronned)) + L1_logdet


if __name__ == "__main__":

    K = [np.random.randn(1000, 400) for i in range(2)]
    c = np.random.randn(400 ** 2, 1)

    r1 = kvs_dot_vec_memhungry(K, c)
    r2 = kvs_dot_vec_loop(K, c)
    r3 = kvs_dot_vec_specialfirst(K, c)
