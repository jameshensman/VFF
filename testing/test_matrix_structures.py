import numpy as np
import tensorflow as tf
from matrix_structures import DiagMat, Rank1Mat, LowRankMat, BlockDiagMat
from unittest import TestCase

np.random.seed(0)


class TestDiagMats(TestCase):
    def setUp(self):
        self.N = 20
        self.d_np = np.random.rand(self.N) + 0.1
        self.A_np = np.diag(self.d_np)
        self.d_tf = tf.placeholder(tf.float64)
        self.mat = DiagMat(self.d_tf)
        self.session = tf.Session()
        self.feed = {self.d_tf: self.d_np}

    def test_get(self):
        A_tf = self.session.run(self.mat.get(), self.feed)
        self.assertTrue(np.allclose(A_tf, self.A_np))

    def test_logdet(self):
        logdet_tf = self.session.run(self.mat.logdet(), self.feed)
        self.assertTrue(np.allclose(np.linalg.slogdet(self.A_np)[1], logdet_tf))

    def test_matmul(self):
        B = np.random.randn(self.N, self.N + 2)
        AB_tf = self.session.run(self.mat.matmul(B), self.feed)
        self.assertTrue(np.allclose(AB_tf, np.dot(self.A_np, B)))

    def test_solve(self):
        B = np.random.randn(self.N, self.N + 2)
        AiB_tf = self.session.run(self.mat.solve(B), self.feed)
        self.assertTrue(np.allclose(AiB_tf, np.linalg.solve(self.A_np, B)))

    def test_trace_KiX(self):
        B = np.random.randn(self.N, self.N)
        tr_AiB_tf = self.session.run(self.mat.trace_KiX(B), self.feed)
        self.assertTrue(np.allclose(tr_AiB_tf, np.trace(np.linalg.solve(self.A_np, B))))

    def test_trace_KiX_against_solve(self):
        B = np.random.randn(self.N, self.N)
        tr_AiB_tf = self.session.run(self.mat.trace_KiX(B), self.feed)
        tr_AiB_tf2 = self.session.run(tf.reduce_sum(tf.diag_part(self.mat.solve(B))), self.feed)
        self.assertTrue(np.allclose(tr_AiB_tf, tr_AiB_tf2))

    def test_get_diag(self):
        d_tf = self.session.run(self.mat.get_diag(), self.feed)
        self.assertTrue(np.allclose(np.diag(self.A_np), d_tf))

    def test_inv_diag(self):
        di_tf = self.session.run(self.mat.inv_diag(), self.feed)
        self.assertTrue(np.allclose(di_tf, np.diag(np.linalg.inv(self.A_np))))


class TestR1Mat(TestDiagMats):
    def setUp(self):
        self.N = 20
        self.d_np = np.random.rand(self.N) + 0.1
        self.v_np = np.random.randn(self.N)
        self.A_np = np.diag(self.d_np) + self.v_np[:, None] * self.v_np[None, :]
        self.d_tf = tf.placeholder(tf.float64)
        self.v_tf = tf.placeholder(tf.float64)
        self.mat = Rank1Mat(self.d_tf, self.v_tf)
        self.session = tf.Session()
        self.feed = {self.d_tf: self.d_np, self.v_tf: self.v_np}


class TestLRMat(TestDiagMats):
    def setUp(self):
        self.N = 20
        self.d_np = np.random.rand(self.N) + 0.1
        self.W_np = np.random.randn(self.N, 2)
        self.A_np = np.diag(self.d_np) + np.dot(self.W_np, self.W_np.T)
        self.d_tf = tf.placeholder(tf.float64)
        self.W_tf = tf.placeholder(tf.float64)
        self.mat = LowRankMat(self.d_tf, self.W_tf)
        self.session = tf.Session()
        self.feed = {self.d_tf: self.d_np, self.W_tf: self.W_np}


class TestBlockMat(TestDiagMats):
    def setUp(self):
        self.N1, self.N2 = 20, 30
        self.N = self.N1 + self.N2
        self.d1_np, self.d2_np = np.random.rand(self.N1) + 0.1, np.random.rand(self.N2) + 0.1
        self.W1_np = np.random.randn(self.N1, 2)
        self.W2_np = np.random.randn(self.N2, 2)
        self.A1_np = np.diag(self.d1_np) + np.dot(self.W1_np, self.W1_np.T)
        self.A2_np = np.diag(self.d2_np) + np.dot(self.W2_np, self.W2_np.T)
        self.A_np = np.vstack(
            [
                np.hstack([self.A1_np, np.zeros([self.N1, self.N2])]),
                np.hstack([np.zeros([self.N2, self.N1]), self.A2_np]),
            ]
        )
        self.d1_tf = tf.placeholder(tf.float64)
        self.W1_tf = tf.placeholder(tf.float64)
        self.d2_tf = tf.placeholder(tf.float64)
        self.W2_tf = tf.placeholder(tf.float64)
        self.mat = BlockDiagMat(
            LowRankMat(self.d1_tf, self.W1_tf), LowRankMat(self.d2_tf, self.W2_tf)
        )
        self.session = tf.Session()
        self.feed = {
            self.d1_tf: self.d1_np,
            self.W1_tf: self.W1_np,
            self.d2_tf: self.d2_np,
            self.W2_tf: self.W2_np,
        }
