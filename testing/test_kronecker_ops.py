import tensorflow as tf
import numpy as np
from unittest import TestCase
from VFF import kronecker_ops
from VFF.matrix_structures import LowRankMat
from functools import reduce


class TestKrons(TestCase):
    def setUp(self):
        self.M = 5
        self.D = 3
        self.ds_np = [np.random.rand(self.M) + 0.1 for i in range(self.D)]
        self.Ws_np = [np.random.randn(self.M, 2) for i in range(self.D)]
        self.Ks_np = [np.diag(d) + np.dot(W, W.T) for d, W in zip(self.ds_np, self.Ws_np)]
        self.ds_tf = [tf.convert_to_tensor(d) for d in self.ds_np]
        self.Ws_tf = [tf.convert_to_tensor(W) for W in self.Ws_np]
        self.Ks_tf = [LowRankMat(d, W) for d, W in zip(self.ds_tf, self.Ws_tf)]

        self.K_np = self.Ks_np[0] * 1
        for ki in self.Ks_np[1:]:
            self.K_np = np.kron(self.K_np, ki)

    def test_kron_vec_mul(self):
        v = np.random.randn(self.M ** self.D, 1)
        Ks = [K.get() for K in self.Ks_tf]
        res_tf = kronecker_ops.kron_vec_mul(Ks, v)
        res_np = np.dot(self.K_np, v)
        np.testing.assert_allclose(res_tf, res_np)

    def test_kron_mat_mul(self):
        mat = np.random.randn(self.M ** self.D, 3)
        Ks = [K.get() for K in self.Ks_tf]
        res_tf = kronecker_ops.kron_mat_mul(Ks, mat, 3)
        res_np = np.dot(self.K_np, mat)
        np.testing.assert_allclose(res_tf, res_np)

    def test_kron_vec_apply_solve(self):
        B = np.random.randn(self.M ** self.D, 1)
        res_np = np.linalg.solve(self.K_np, B)
        res_tf = kronecker_ops.kron_vec_apply(self.Ks_tf, B, "solve")
        np.testing.assert_allclose(res_tf, res_np)


class TestKVS(TestCase):
    def setUp(self):
        self.M = 5
        self.N = 20
        self.D = 3
        self.Ks_np = [np.random.randn(self.N, self.M) for i in range(self.D)]
        self.Ks_tf = [tf.convert_to_tensor(K) for K in self.Ks_np]

        rows = [
            np.kron(np.kron(self.Ks_np[0][i], self.Ks_np[1][i]), self.Ks_np[2][i])
            for i in range(self.N)
        ]
        self.K_full = np.vstack(rows)

    def test_make_kvs(self):
        tf_res = kronecker_ops.make_kvs(self.Ks_tf)
        np.testing.assert_allclose(tf_res, self.K_full)

    def test_kvs_dot_vec(self):
        B = np.random.randn(self.M ** self.D, 1)
        tf_res = kronecker_ops.kvs_dot_vec(self.Ks_tf, B)
        np_res = np.dot(self.K_full, B)
        np.testing.assert_allclose(tf_res, np_res)


def test_logdetsum_np():
    Ls = [np.tril(np.random.randn(3, 3)) for i in range(3)]
    Rs = [np.tril(np.random.randn(3, 3)) for i in range(3)]

    actual = kronecker_ops.log_det_kron_sum_np(Ls, Rs)

    LLs = [np.dot(L, L.T) for L in Ls]
    RRs = [np.dot(R, R.T) for R in Rs]
    _, expected = np.linalg.slogdet(reduce(np.kron, LLs) + reduce(np.kron, RRs))

    np.testing.assert_allclose(actual, expected)


def test_logdetsum_tf():
    Ls_np = [np.tril(np.random.randn(3, 3)) for i in range(3)]
    Rs_np = [np.tril(np.random.randn(3, 3)) for i in range(3)]
    Ls_tf = [tf.convert_to_tensor(L) for L in Ls_np]
    Rs_tf = [tf.convert_to_tensor(R) for R in Rs_np]

    res_np = kronecker_ops.log_det_kron_sum_np(Ls_np, Rs_np)
    res_tf = kronecker_ops.log_det_kron_sum(Ls_tf, Rs_tf)
    np.testing.assert_allclose(res_tf, res_np)
