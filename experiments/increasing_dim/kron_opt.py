import GPflow
import numpy as np
from time import time
import pandas as pd

from config import dimensions, noise_var, prodkern, repeats, num_freqs
import VFF

results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        data = np.load('data_dim{}_rep{}.npz'.format(dim, r))
        for M in num_freqs:
            if (M ** dim) > 1e5:
                continue
            # build kronecker GP model
            a, b = -0.5 * np.ones(dim), 1.5 * np.ones(dim)
            m = VFF.vgp.VGP_kron(data['Xtrain'], data['Ytrain'], np.arange(M), a, b,
                                 kerns=prodkern(dim).kern_list,
                                 likelihood=GPflow.likelihoods.Gaussian(),
                                 use_two_krons=True)
            m.likelihood.variance = noise_var

            # only optimize q(u)
            m.kerns.fixed = True
            m.likelihood.fixed = True

            start = time()
            m.optimize()
            marg_lik = m.compute_log_likelihood().squeeze()
            mean_log_pred = np.mean(m.predict_density(data['Xtest'], data['Ytest']))
            t = time() - start

            results = results.append(dict(dim=dim, rep=r, marg_lik=marg_lik,
                                          mean_log_pred=mean_log_pred, time=t,
                                          num_inducing=M),
                                     ignore_index=True)

            # do this inside the loop so we can get partial results if something crashes
            results.to_csv('kron_opt.csv')
