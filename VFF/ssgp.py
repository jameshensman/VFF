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


import GPflow
import numpy as np
import tensorflow as tf


class SSGP(GPflow.model.GPModel):
    """
    The Sparse Spectrum GP, judiciously copied from Miguel Lazaro Gredilla's
    MATLAB code, available at http://www.tsc.uc3m.es/~miguel/downloads.php. His
    code remains as comments in this file.
    """

    def __init__(self, X, Y, kern, num_basis=10):
        lik = GPflow.likelihoods.Gaussian()
        mf = GPflow.mean_functions.Zero()
        GPflow.model.GPModel.__init__(self, X, Y, kern=kern, likelihood=lik, mean_function=mf)
        input_dim = self.X.shape[1]
        if isinstance(kern, GPflow.kernels.RBF):
            self.omega = GPflow.param.Param(np.random.randn(num_basis, input_dim))
        elif isinstance(kern, GPflow.kernels.Matern12):
            self.omega = GPflow.param.Param(np.random.standard_cauchy((num_basis, input_dim)))
        elif isinstance(kern, GPflow.kernels.Matern32):
            self.omega = GPflow.param.Param(np.random.standard_t(2, (num_basis, input_dim)))
        elif isinstance(kern, GPflow.kernels.Matern52):
            self.omega = GPflow.param.Param(np.random.standard_t(3, (num_basis, input_dim)))
        else:
            raise NotImplementedError
        assert self.Y.shape[1] == 1
        self.num_latent = 1

        # m=(length(optimizeparams)-D-2)/D;                                       % number of basis
        # ell  = exp(optimizeparams(1:D));                                        % characteristic lengthscale
        # sf2  = exp(2*optimizeparams(D+1));                                      % signal power
        # sn2  = exp(2*optimizeparams(D+2));                                      % noise power
        # w = reshape(optimizeparams(D+3:end), [m, D]);                           % unscaled model angular frequencies

    def build_likelihood(self):

        # w = w./repmat(ell',[m,1]);                                              % scaled model angular frequencies
        w = self.omega / self.kern.lengthscales
        m = tf.shape(self.omega)[0]
        m_float = tf.cast(m, tf.float64)

        # phi = x_tr*w';
        phi = tf.matmul(self.X, tf.transpose(w))
        # phi = [cos(phi) sin(phi)];                                              % design matrix
        phi = tf.concat([tf.cos(phi), tf.sin(phi)], axis=1)

        # R = chol((sf2/m)*(phi'*phi) + sn2*eye(2*m));                            % calculate some often-used constants
        A = (self.kern.variance / m_float) * tf.matmul(tf.transpose(phi), phi)\
            + self.likelihood.variance * GPflow.tf_wraps.eye(2*m)
        RT = tf.cholesky(A)
        R = tf.transpose(RT)

        # PhiRiphi/R;
        # RtiPhit = PhiRi';
        RtiPhit = tf.matrix_triangular_solve(RT, tf.transpose(phi))
        # Rtiphity=RtiPhit*y_tr;
        Rtiphity = tf.matmul(RtiPhit, self.Y)

        # % output NLML
        # out1=0.5/sn2*(sum(y_tr.^2)-sf2/m*sum(Rtiphity.^2))+ ...
        out = 0.5/self.likelihood.variance*(tf.reduce_sum(tf.square(self.Y)) -
                                            self.kern.variance/m_float*tf.reduce_sum(tf.square(Rtiphity)))
        # +sum(log(diag(R)))+(n/2-m)*log(sn2)+n/2*log(2*pi);
        n = tf.cast(tf.shape(self.X)[0], tf.float64)
        out += tf.reduce_sum(tf.log(tf.diag_part(R)))\
            + (n/2.-m_float) * tf.log(self.likelihood.variance)\
            + n/2*np.log(2*np.pi)
        return -out

    def build_predict(self, Xnew, full_cov=False):
        # w = w./repmat(ell',[m,1]);                                              % scaled model angular frequencies
        w = self.omega / self.kern.lengthscales
        m = tf.shape(self.omega)[0]
        m_float = tf.cast(m, tf.float64)

        # phi = x_tr*w';
        phi = tf.matmul(self.X, tf.transpose(w))
        # phi = [cos(phi) sin(phi)];                                              % design matrix
        phi = tf.concat([tf.cos(phi), tf.sin(phi)], axis=1)

        # R = chol((sf2/m)*(phi'*phi) + sn2*eye(2*m));                            % calculate some often-used constants
        A = (self.kern.variance / m_float) * tf.matmul(tf.transpose(phi), phi)\
            + self.likelihood.variance * GPflow.tf_wraps.eye(2*m)
        RT = tf.cholesky(A)
        R = tf.transpose(RT)

        # RtiPhit = PhiRi';
        RtiPhit = tf.matrix_triangular_solve(RT, tf.transpose(phi))
        # Rtiphity=RtiPhit*y_tr;
        Rtiphity = tf.matmul(RtiPhit, self.Y)

        # alfa=sf2/m*(R\Rtiphity);                                                % cosines/sines coefficients
        alpha = self.kern.variance / m_float * tf.matrix_triangular_solve(R, Rtiphity, lower=False)

        # phistar = x_tst*w';
        phistar = tf.matmul(Xnew, tf.transpose(w))
        # phistar = [cos(phistar) sin(phistar)];                              % test design matrix
        phistar = tf.concat([tf.cos(phistar), tf.sin(phistar)], axis=1)
        # out1(beg_chunk:end_chunk) = phistar*alfa;                           % Predictive mean
        mean = tf.matmul(phistar, alpha)

        # % also output predictive variance
        # out2(beg_chunk:end_chunk) = sn2*(1+sf2/m*sum((phistar/R).^2,2));% Predictive variance
        RtiPhistart = tf.matrix_triangular_solve(RT, tf.transpose(phistar))
        PhiRistar = tf.transpose(RtiPhistart)
        # NB: do not add in noise variance to the predictive var: GPflow does that for us.
        if full_cov:
            var = self.likelihood.variance * self.kern.variance / m_float *\
                tf.matmul(PhiRistar, tf.transpose(PhiRistar)) + \
                GPflow.tf_wraps.eye(tf.shape(Xnew)[0]) * 1e-6
            var = tf.expand_dims(var, 2)
        else:
            var = self.likelihood.variance * self.kern.variance / m_float * tf.reduce_sum(tf.square(PhiRistar), 1)
            var = tf.expand_dims(var, 1)

        return mean, var
