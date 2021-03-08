import tensorflow as tf
import numpy as np
from unittest import TestCase
import kronecker_ops
from matrix_structures import LowRankMat
from functools import reduce


class TestKrons(TestCase):
    def setUp(self):
        self.M = 5
        self.D = 3
        self.ds_np = [np.random.rand(self.M) + 0.1 for i in range(self.D)]
        self.Ws_np = [np.random.randn(self.M, 2) for i in range(self.D)]
        self.Ks_np = [np.diag(d) + np.dot(W, W.T) for d, W in zip(self.ds_np, self.Ws_np)]

        self.K_np = self.Ks_np[0] * 1
        for ki in self.Ks_np[1:]:
            self.K_np = np.kron(self.K_np, ki)

        self.ds_tf = [tf.placeholder(tf.float64, [self.M]) for i in range(self.D)]
        self.Ws_tf = [tf.placeholder(tf.float64, [self.M, 2]) for i in range(self.D)]
        self.Ks_tf = [LowRankMat(d, W) for d, W in zip(self.ds_tf, self.Ws_tf)]

        self.session = tf.Session()
        self.feed = dict(zip(self.ds_tf + self.Ws_tf, self.ds_np + self.Ws_np))

    def test_kron_vec_mul(self):
        v = np.random.randn(self.M ** self.D, 1)
        Ks = [K.get() for K in self.Ks_tf]
        res_tf = self.session.run(kronecker_ops.kron_vec_mul(Ks, v), self.feed)
        res_np = np.dot(self.K_np, v)
        self.assertTrue(np.allclose(res_tf, res_np))

    def test_kron_mat_mul(self):
        mat = np.random.randn(self.M ** self.D, 3)
        Ks = [K.get() for K in self.Ks_tf]
        res_tf = self.session.run(kronecker_ops.kron_mat_mul(Ks, mat, 3), self.feed)
        res_np = np.dot(self.K_np, mat)
        self.assertTrue(np.allclose(res_tf, res_np))

    def test_kron_vec_apply_solve(self):
        B = np.random.randn(self.M ** self.D, 1)
        res_np = np.linalg.solve(self.K_np, B)
        res_tf = self.session.run(kronecker_ops.kron_vec_apply(self.Ks_tf, B, "solve"), self.feed)
        self.assertTrue(np.allclose(res_tf, res_np))


class TestKVS(TestCase):
    def setUp(self):
        self.M = 5
        self.N = 20
        self.D = 3
        self.Ks_np = [np.random.randn(self.N, self.M) for i in range(self.D)]

        rows = [
            np.kron(np.kron(self.Ks_np[0][i], self.Ks_np[1][i]), self.Ks_np[2][i])
            for i in range(self.N)
        ]
        self.K_full = np.vstack(rows)

        self.Ks_tf = [tf.placeholder(tf.float64, [self.N, self.M]) for i in range(self.D)]

        self.session = tf.Session()
        self.feed = dict(zip(self.Ks_tf, self.Ks_np))

    def test_make_kvs(self):
        tf_res = self.session.run(kronecker_ops.make_kvs(self.Ks_tf), self.feed)
        self.assertTrue(np.allclose(tf_res, self.K_full))

    def test_kvs_dot_vec(self):
        B = np.random.randn(self.M ** self.D, 1)
        tf_res = self.session.run(kronecker_ops.kvs_dot_vec(self.Ks_tf, B), self.feed)
        np_res = np.dot(self.K_full, B)
        self.assertTrue(np.allclose(tf_res, np_res))


class TestLogDetSum_np(TestCase):
    def setUp(self):
        self.Ls = [np.tril(np.random.randn(3, 3)) for i in range(3)]
        self.Rs = [np.tril(np.random.randn(3, 3)) for i in range(3)]

    def test_hard(self):
        LLs = [np.dot(L, L.T) for L in self.Ls]
        RRs = [np.dot(R, R.T) for R in self.Rs]
        _, result1 = np.linalg.slogdet(reduce(np.kron, LLs) + reduce(np.kron, RRs))
        self.assertTrue(np.allclose(result1, kronecker_ops.log_det_kron_sum_np(self.Ls, self.Rs)))


class TestLogDetSum_tf(TestCase):
    def setUp(self):
        self.Ls_np = [np.tril(np.random.randn(3, 3)) for i in range(3)]
        self.Rs_np = [np.tril(np.random.randn(3, 3)) for i in range(3)]
        self.Ls_tf = [tf.placeholder(tf.float64, [None, None]) for i in range(3)]
        self.Rs_tf = [tf.placeholder(tf.float64, [None, None]) for i in range(3)]

        self.feed = dict(zip(self.Ls_tf + self.Rs_tf, self.Ls_np + self.Rs_np))
        self.session = tf.Session()

    def test_hard(self):
        res_np = kronecker_ops.log_det_kron_sum_np(self.Ls_np, self.Rs_np)
        res_tf = self.session.run(kronecker_ops.log_det_kron_sum(self.Ls_tf, self.Rs_tf), self.feed)
        self.assertTrue(np.allclose(res_np, res_tf))
