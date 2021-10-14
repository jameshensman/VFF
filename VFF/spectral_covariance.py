# Copyright 2021 ST John
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


import numpy as np
import gpflow
import tensorflow as tf
from .matrix_structures import DiagMat, Rank1Mat, LowRankMat, BlockDiagMat
from gpflow import default_float


def make_Kuu(kern, a, b, ms):
    """
    # Make a representation of the Kuu matrices
    """
    omegas = 2.0 * np.pi * ms / (b - a)
    if default_float() is np.float32:
        omegas = omegas.astype(np.float32)
    if isinstance(kern, gpflow.kernels.Matern12):
        # cos part first
        lamb = 1.0 / kern.lengthscales
        two_or_four = np.where(omegas == 0, 2.0, 4.0)
        d_cos = (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kern.variance / two_or_four
        v_cos = tf.ones(tf.shape(d_cos), default_float()) / tf.sqrt(kern.variance)

        # now the sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        d_sin = (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kern.variance / 4.0

        return BlockDiagMat(Rank1Mat(d_cos, v_cos), DiagMat(d_sin))

    elif isinstance(kern, gpflow.kernels.Matern32):
        # cos part first
        lamb = np.sqrt(3.0) / kern.lengthscales
        four_or_eight = np.where(omegas == 0, 4.0, 8.0)
        d_cos = (
            (b - a)
            * tf.square(tf.square(lamb) + tf.square(omegas))
            / tf.pow(lamb, 3)
            / kern.variance
            / four_or_eight
        )
        v_cos = tf.ones(tf.shape(d_cos), default_float()) / tf.sqrt(kern.variance)

        # now the sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        d_sin = (
            (b - a)
            * tf.square(tf.square(lamb) + tf.square(omegas))
            / tf.pow(lamb, 3)
            / kern.variance
            / 8.0
        )
        v_sin = omegas / lamb / tf.sqrt(kern.variance)
        return BlockDiagMat(Rank1Mat(d_cos, v_cos), Rank1Mat(d_sin, v_sin))

    elif isinstance(kern, gpflow.kernels.Matern52):
        # cos part:
        lamb = np.sqrt(5.0) / kern.lengthscales
        sixteen_or_32 = np.where(omegas == 0, 16.0, 32.0)
        v1 = (3 * tf.square(omegas / lamb) - 1) / tf.sqrt(8 * kern.variance)
        v2 = tf.ones(tf.shape(v1), default_float()) / tf.sqrt(kern.variance)
        W_cos = tf.concat([tf.expand_dims(v1, 1), tf.expand_dims(v2, 1)], axis=1)
        d_cos = (
            3
            * (b - a)
            / sixteen_or_32
            / tf.pow(lamb, 5)
            / kern.variance
            * tf.pow(tf.square(lamb) + tf.square(omegas), 3)
        )

        # sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        v_sin = np.sqrt(3.0) * omegas / lamb / tf.sqrt(kern.variance)
        d_sin = (
            3
            * (b - a)
            / 32.0
            / tf.pow(lamb, 5)
            / kern.variance
            * tf.pow(tf.square(lamb) + tf.square(omegas), 3)
        )
        return BlockDiagMat(LowRankMat(d_cos, W_cos), Rank1Mat(d_sin, v_sin))
    else:
        raise NotImplementedError


def make_Kuf_no_edges(X, a, b, ms):
    omegas = 2.0 * np.pi * ms / (b - a)
    Kuf_cos = tf.transpose(tf.cos(omegas * (X - a)))
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas * (X - a)))
    return tf.concat([Kuf_cos, Kuf_sin], axis=0)


def make_Kuf(k, X, a, b, ms):
    omegas = 2.0 * np.pi * ms / (b - a)
    if default_float() is np.float32:
        omegas = omegas.astype(np.float32)
    Kuf_cos = tf.transpose(tf.cos(omegas * (X - a)))
    omegas_sin = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas_sin * (X - a)))

    # correct Kfu outside [a, b]
    lt_a_sin = tf.tile(tf.transpose(X) < a, [len(ms) - 1, 1])
    gt_b_sin = tf.tile(tf.transpose(X) > b, [len(ms) - 1, 1])
    lt_a_cos = tf.tile(tf.transpose(X) < a, [len(ms), 1])
    gt_b_cos = tf.tile(tf.transpose(X) > b, [len(ms), 1])
    if isinstance(k, gpflow.kernels.Matern12):
        # Kuf_sin[:, np.logical_or(X.flatten() < a, X.flatten() > b)] = 0
        Kuf_sin = tf.where(
            tf.logical_or(lt_a_sin, gt_b_sin), tf.zeros(tf.shape(Kuf_sin), default_float()), Kuf_sin
        )
        Kuf_cos = tf.where(
            lt_a_cos,
            tf.tile(tf.exp(-tf.abs(tf.transpose(X - a)) / k.lengthscales), [len(ms), 1]),
            Kuf_cos,
        )
        Kuf_cos = tf.where(
            gt_b_cos,
            tf.tile(tf.exp(-tf.abs(tf.transpose(X - b)) / k.lengthscales), [len(ms), 1]),
            Kuf_cos,
        )
    elif isinstance(k, gpflow.kernels.Matern32):
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = tf.tile((1 + arg) * tf.exp(-arg), [len(ms), 1])
        Kuf_cos = tf.where(lt_a_cos, edge, Kuf_cos)
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - b) / k.lengthscales
        edge = tf.tile((1 + arg) * tf.exp(-arg), [len(ms), 1])
        Kuf_cos = tf.where(gt_b_cos, edge, Kuf_cos)

        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = (tf.transpose(X) - a) * tf.exp(-arg) * omegas_sin[:, None]
        Kuf_sin = tf.where(lt_a_sin, edge, Kuf_sin)
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - b) / k.lengthscales
        edge = (tf.transpose(X) - b) * tf.exp(-arg) * omegas_sin[:, None]
        Kuf_sin = tf.where(gt_b_sin, edge, Kuf_sin)
    elif isinstance(k, gpflow.kernels.Matern52):
        # edges not implemented yet
        tf.debugging.assert_greater_equal(
            X,
            a,
            message="Edges not implemented for Matern52",
            name="assert_left_edge",
        )
        tf.debugging.assert_less_equal(
            X,
            b,
            message="Edges not implemented for Matern52",
            name="assert_right_edge",
        )
    else:
        raise NotImplementedError
    return tf.concat([Kuf_cos, Kuf_sin], axis=0)


def make_Kuf_np(X, a, b, ms):
    omegas = 2.0 * np.pi * ms / (b - a)
    Kuf_cos = np.cos(omegas * (X - a)).T
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = np.sin(omegas * (X - a)).T
    return np.vstack([Kuf_cos, Kuf_sin])


def make_Kuf_np_with_edges(k, X, a, b, ms):
    omegas = 2.0 * np.pi * ms / (b - a)
    Kuf_cos = np.cos(omegas * (X - a)).T
    omegas_sin = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = np.sin(omegas_sin * (X - a)).T

    # correct Kfu outside [a, b]
    if isinstance(k, gpflow.kernels.Matern12):
        Kuf_sin[:, np.logical_or(X.flatten() < a, X.flatten() > b)] = 0
        X_a = X[X.flatten() < a].T
        X_b = X[X.flatten() > b].T
        Kuf_cos[:, X.flatten() < a] = np.exp(-np.abs(X_a - a) / k.lengthscales.value)
        Kuf_cos[:, X.flatten() > b] = np.exp(-np.abs(X_b - b) / k.lengthscales.value)

    elif isinstance(k, gpflow.kernels.Matern32):
        X_a = X[X.flatten() < a].T
        X_b = X[X.flatten() > b].T

        arg = np.sqrt(3) * np.abs(X_a - a) / k.lengthscales.value
        Kuf_cos[:, X.flatten() < a] = (1 + arg) * np.exp(-arg)
        arg = np.sqrt(3) * np.abs(X_b - b) / k.lengthscales.value
        Kuf_cos[:, X.flatten() > b] = (1 + arg) * np.exp(-arg)

        arg = np.sqrt(3) * np.abs(a - X_a) / k.lengthscales.value
        Kuf_sin[:, X.flatten() < a] = (X_a - a) * np.exp(-arg) * omegas_sin[:, None]
        arg = np.sqrt(3) * np.abs(X_b - b) / k.lengthscales.value
        Kuf_sin[:, X.flatten() > b] = (X_b - b) * np.exp(-arg) * omegas_sin[:, None]

    elif isinstance(k, gpflow.kernels.Matern52):
        assert not (np.any(X < a) or np.any(X > b))
        raise NotImplementedError
    else:
        raise NotImplementedError

    return np.vstack([Kuf_cos, Kuf_sin])
