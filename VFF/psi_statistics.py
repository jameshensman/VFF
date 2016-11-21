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
import numpy as np


def psi1(mean, var, a, b, ms):

    # this bit is the same as Kuf on the mean
    omegas = 2. * np.pi * ms / (b - a)
    Kuf_cos = tf.transpose(tf.cos(omegas * (mean - a)))
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas * (mean - a)))

    # this bit depends on the variance
    a = tf.transpose(tf.exp(- tf.square(omegas) * var / 2))
    Psi1_cos = Kuf_cos * a
    Psi1_sin = Kuf_sin * a

    return tf.concat(0, [Psi1_cos, Psi1_sin])


def psi2(mean, var, a, b, ms):
    # mean and var must be N x 1

    # this bit is the same as Kuf on the mean
    omegas = 2. * np.pi * ms / (b - a)
    omegas_add = tf.reshape(omegas, (-1, 1)) + tf.reshape(omegas, (1, -1))
    omegas_diff = tf.reshape(omegas, (-1, 1)) - tf.reshape(omegas, (1, -1))
    omegas_add = tf.expand_dims(omegas_add, 0)
    omegas_diff = tf.expand_dims(omegas_diff, 0)

    exp_add = tf.exp(-tf.expand_dims(var, 1) * tf.square(omegas_add) / 2)
    exp_diff = tf.exp(-var * tf.square(omegas_diff) / 2)

    # TODO


def uniform(a, b, ms, low, up):
    # here's the cosine part.
    omegas_cos = 2. * np.pi * ms / (b-a)
    w = omegas_cos.reshape(-1, 1)
    m = omegas_cos.reshape(1, -1)
    # integral_a^b cos(w x) cos(m x) dx = (-m sin(a m) cos(a w)+w cos(a m) sin(a w)+m sin(b m) cos(b w)-w cos(b m) sin(b w))/(m^2-w^2)
    coscos = (- m * tf.sin(low * m) * tf.cos(low * w)
              + w * tf.cos(low * m) * tf.sin(low * w)
              + m * tf.sin(up * m) * tf.cos(up * w)
              - w * tf.cos(up * m) * tf.sin(up * w))/(tf.square(m)-tf.square(w))

    # integral_a^b cos^2(w x) dx = (2 w (b-a)-sin(2 a w)+sin(2 b w))/(4 w)
    cos2 = (2 * w * (up - low) - tf.sin(2 * low * w) + tf.sin(2 * up * w)) / (4 * w)

    # here's the sin part

    omegas_sin = omegas_cos[omegas_cos != 0]  # don't compute omega=0
    w = omegas_sin.reshape(-1, 1)
    m = omegas_sin.reshape(1, -1)
    # integral_a^b sin(w x) sin(m x) dx = (-w sin(a m) cos(a w)+m cos(a m) sin(a w)+w sin(b m) cos(b w)-m cos(b m) sin(b w))/(m^2-w^2)
    sinsin = (- w * tf.sin(low * m) * tf.cos(low * w)
              + m * tf.cos(low * m) * tf.sin(low * w)
              + w * tf.sin(up * m) * tf.cos(up * w)
              - m * tf.cos(up * m) * tf.sin(up * w))/(tf.square(m)-tf.square(w))

    # integral_a^b sin^2(w x) dx = (2 w (b-a)+sin(2 a w)-sin(2 b w))/(4 w)
    sin2 = (2 * w * (up - low) + tf.sin(2 * a * w) - tf.sin(2 * up * w)) / (4 * w)

    # here the 'cross' part
    # integral_a^b cos(w x) sin(m x) dx = (w sin(a m) sin(a w)+m cos(a m) cos(a w)-w sin(b m) sin(b w)-m cos(b m) cos(b w))/(m^2-w^2)
    w = omegas_cos.reshape(-1, 1)
    m = omegas_sin.reshape(1, -1)
    sincos = (w * tf.sin(low * m) * tf.sin(low * w)
              + m * tf.cos(low * m) * tf.cos(low * w)
              - w * tf.sin(up * m) * tf.sin(up * w)
              - m * tf.cos(up * m) * tf.cos(up * w))/(tf.square(m) - tf.square(w))

    return coscos, cos2, sinsin, sin2, sincos




    # integral_a^b sin(w x) sin(m x) dx = (-w sin(a m) cos(a w)+m cos(a m) sin(a w)+w sin(b m) cos(b w)-m cos(b m) sin(b w))/(m^2-w^2)
