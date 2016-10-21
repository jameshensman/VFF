import numpy as np
import GPflow
import tensorflow as tf
from .matrix_structures import DiagMat, Rank1Mat, LowRankMat, BlockDiagMat
from GPflow import settings
float_type = settings.dtypes.float_type


def make_Kuu(kern, a, b, ms):
    """
    # Make a representation of the Kuu matrices
    """
    omegas = 2. * np.pi * ms / (b-a)
    if float_type is tf.float32:
        omegas = omegas.astype(np.float32)
    if isinstance(kern, GPflow.kernels.Matern12):
        # cos part first
        lamb = 1./kern.lengthscales
        two_or_four = np.where(omegas == 0, 2., 4.)
        d_cos = (b-a) * (tf.square(lamb) + tf.square(omegas)) \
            / lamb / kern.variance / two_or_four
        v_cos = tf.ones(tf.shape(d_cos), float_type)\
            / tf.sqrt(kern.variance)

        # now the sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        d_sin = (b-a) * (tf.square(lamb) + tf.square(omegas)) \
            / lamb / kern.variance / 4.

        return BlockDiagMat(Rank1Mat(d_cos, v_cos), DiagMat(d_sin))

    elif isinstance(kern, GPflow.kernels.Matern32):
        # cos part first
        lamb = np.sqrt(3.)/kern.lengthscales
        four_or_eight = np.where(omegas == 0, 4., 8.)
        d_cos = (b-a) * tf.square(tf.square(lamb) + tf.square(omegas)) \
            / tf.pow(lamb, 3) / kern.variance / four_or_eight
        v_cos = tf.ones(tf.shape(d_cos), float_type)\
            / tf.sqrt(kern.variance)

        # now the sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        d_sin = (b-a) * tf.square(tf.square(lamb) + tf.square(omegas)) \
            / tf.pow(lamb, 3) / kern.variance / 8.
        v_sin = omegas / lamb / tf.sqrt(kern.variance)
        return BlockDiagMat(Rank1Mat(d_cos, v_cos), Rank1Mat(d_sin, v_sin))

    elif isinstance(kern, GPflow.kernels.Matern52):
        # cos part:
        lamb = np.sqrt(5.0) / kern.lengthscales
        sixteen_or_32 = np.where(omegas == 0, 16., 32.)
        v1 = (3 * tf.square(omegas / lamb) - 1) / tf.sqrt(8 * kern.variance)
        v2 = tf.ones(tf.shape(v1), float_type) / tf.sqrt(kern.variance)
        W_cos = tf.concat(1, [tf.expand_dims(v1, 1), tf.expand_dims(v2, 1)])
        d_cos = 3 * (b - a) / sixteen_or_32 / tf.pow(lamb, 5)\
            / kern.variance\
            * tf.pow(tf.square(lamb) + tf.square(omegas), 3)

        # sin part
        omegas = omegas[omegas != 0]  # don't compute omega=0
        v_sin = np.sqrt(3.) * omegas / lamb / tf.sqrt(kern.variance)
        d_sin = 3 * (b - a) / 32. / tf.pow(lamb, 5) / kern.variance\
            * tf.pow(tf.square(lamb) + tf.square(omegas), 3)
        return BlockDiagMat(LowRankMat(d_cos, W_cos), Rank1Mat(d_sin, v_sin))
    else:
        raise NotImplementedError


def make_Kuf_no_edges(X, a, b, ms):
    omegas = 2. * np.pi * ms / (b-a)
    Kuf_cos = tf.transpose(tf.cos(omegas * (X - a)))
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas * (X - a)))
    return tf.concat(0, [Kuf_cos, Kuf_sin])


def make_Kuf(k, X, a, b, ms):
    omegas = 2. * np.pi * ms / (b-a)
    if float_type is tf.float32:
        omegas = omegas.astype(np.float32)
    Kuf_cos = tf.transpose(tf.cos(omegas * (X - a)))
    omegas_sin = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = tf.transpose(tf.sin(omegas_sin * (X - a)))

    # correct Kfu outside [a, b]
    lt_a_sin = tf.tile(tf.transpose(X) < a, [len(ms)-1, 1])
    gt_b_sin = tf.tile(tf.transpose(X) > b, [len(ms)-1, 1])
    lt_a_cos = tf.tile(tf.transpose(X) < a, [len(ms), 1])
    gt_b_cos = tf.tile(tf.transpose(X) > b, [len(ms), 1])
    if isinstance(k, GPflow.kernels.Matern12):
        # Kuf_sin[:, np.logical_or(X.flatten() < a, X.flatten() > b)] = 0
        Kuf_sin = tf.select(tf.logical_or(lt_a_sin, gt_b_sin), tf.zeros(tf.shape(Kuf_sin), float_type), Kuf_sin)
        Kuf_cos = tf.select(lt_a_cos, tf.tile(tf.exp(-tf.abs(tf.transpose(X-a))/k.lengthscales), [len(ms), 1]), Kuf_cos)
        Kuf_cos = tf.select(gt_b_cos, tf.tile(tf.exp(-tf.abs(tf.transpose(X-b))/k.lengthscales), [len(ms), 1]), Kuf_cos)
    elif isinstance(k, GPflow.kernels.Matern32):
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = tf.tile((1 + arg) * tf.exp(-arg), [len(ms), 1])
        Kuf_cos = tf.select(lt_a_cos, edge, Kuf_cos)
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = tf.tile((1 + arg) * tf.exp(-arg), [len(ms), 1])
        Kuf_cos = tf.select(gt_b_cos, edge, Kuf_cos)

        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - a) / k.lengthscales
        edge = (tf.transpose(X) - a) * tf.exp(-arg) * omegas_sin[:, None]
        Kuf_sin = tf.select(lt_a_sin, edge, Kuf_sin)
        arg = np.sqrt(3) * tf.abs(tf.transpose(X) - b) / k.lengthscales
        edge = (tf.transpose(X) - b) * tf.exp(-arg) * omegas_sin[:, None]
        Kuf_sin = tf.select(gt_b_sin, edge, Kuf_sin)
    return tf.concat(0, [Kuf_cos, Kuf_sin])


def make_Kuf_np(X, a, b, ms):
    omegas = 2. * np.pi * ms / (b-a)
    Kuf_cos = np.cos(omegas * (X - a)).T
    omegas = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = np.sin(omegas * (X - a)).T
    return np.vstack([Kuf_cos, Kuf_sin])


def make_Kuf_np_with_edges(k, X, a, b, ms):
    omegas = 2. * np.pi * ms / (b-a)
    Kuf_cos = np.cos(omegas * (X - a)).T
    omegas_sin = omegas[omegas != 0]  # don't compute zeros freq.
    Kuf_sin = np.sin(omegas_sin * (X - a)).T

    # correct Kfu outside [a, b]
    if isinstance(k, GPflow.kernels.Matern12):
        Kuf_sin[:, np.logical_or(X.flatten() < a, X.flatten() > b)] = 0
        X_a = X[X.flatten() < a].T
        X_b = X[X.flatten() > b].T
        Kuf_cos[:, X.flatten() < a] = np.exp(-np.abs(X_a-a)/k.lengthscales.value)
        Kuf_cos[:, X.flatten() > b] = np.exp(-np.abs(X_b-b)/k.lengthscales.value)

    elif isinstance(k, GPflow.kernels.Matern32):
        X_a = X[X.flatten() < a].T
        X_b = X[X.flatten() > b].T

        arg = np.sqrt(3) * np.abs(X_a - a) / k.lengthscales.value
        Kuf_cos[:, X.flatten() < a] = (1 + arg) * np.exp(-arg)
        arg = np.sqrt(3) * np.abs(X_b - b) / k.lengthscales.value
        Kuf_cos[:, X.flatten() > b] = (1 + arg) * np.exp(-arg)

        arg = np.sqrt(3) * np.abs(a - X_a) / k.lengthscales.value
        Kuf_sin[:, X.flatten() < a] = (X_a - a) * np.exp(-arg) * omegas_sin[:, None]
        arg = np.sqrt(3) * np.abs(X_b - b) / k.lengthscales.value
        Kuf_sin[:, X.flatten() > b] = (X_b-b) * np.exp(-arg) * omegas_sin[:, None]

    elif isinstance(k, GPflow.kernels.Matern52):
        X_a = X[X.flatten() < a].T
        X_b = X[X.flatten() > b].T

        # arg = np.sqrt(5) * np.abs(X_a - a) / k.lengthscales.value
        # Kuf_cos[:, X.flatten() < a] = (1 + arg) * np.exp(-arg)
        # arg = np.sqrt(3) * np.abs(X_b - b) / k.lengthscales.value
        # Kuf_cos[:, X.flatten() > b] = (1 + arg) * np.exp(-arg)

        # arg = np.sqrt(5) * np.abs(a - X_a) / k.lengthscales.value
        # Kuf_sin[:, X.flatten() < a] = (X_a - a) * np.exp(-arg) * omegas_sin[:, None]
        arg = np.sqrt(5) * np.abs(X_b - b) / k.lengthscales.value
        Kuf_sin[:, X.flatten() > b] = (3./5.) / k.lengthscales.value ** 2 * omegas_sin[:, None] * np.exp(-arg) *\
            ((3./5.)/k.lengthscales.value**2 * (X_b - b) + 5 * np.sqrt(5) / 3. / k.lengthscales.value**3 * (X_b - b)**2)

    else:
        raise NotImplementedError
    return np.vstack([Kuf_cos, Kuf_sin])
