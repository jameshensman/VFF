import numpy as np
import sys
import gpflow
import VFF

from time import time

from config import *

dim = sys.argv[1]
rep = sys.argv[2]

print("vff: dimension {}, replicate {}".format(dim, r))

# data
data = np.load("data/data_dim{}_rep{}.npz".format(dim, 0))

# full_gp
def prodkern(dim):
    return gpflow.kernels.Prod(
        [gpflow.kernels.Matern32(1, active_dims=[i], lengthscales=lengthscale) for i in range(dim)]
    )


k = prodkern(dim)
m = gpflow.gpr.GPR(data["Xtrain"], data["Ytrain"], kern=k)
m.likelihood.variance = noise_var
data = np.load("data/data_dim{}_rep{}.npz".format(dim, r))
marg_lik = m.compute_log_likelihood().squeeze()
mean_log_pred = np.mean(m.predict_density(data["Xtest"], data["Ytest"]))

file = open("results/full.csv", "a")
file.write("{}, {}, {}, {}".format(dim, rep, marg_lik, mean_log_pred))
file.close()


##########################
# kron
results = pd.DataFrame()

for dim in dimensions:
    a, b = -1.5 * np.ones(dim), 1.5 * np.ones(dim)
    k = prodkern(dim)
    for r in range(repeats):
        print("kron replicate ", r, "/", repeats)
        data = np.load("data/data_dim{}_rep{}.npz".format(dim, r))
        for M in num_freqs:
            if (2 * M - 1) ** dim:
                a, b = -0.5 * np.ones(dim), 1.5 * np.ones(dim)
                m = VFF.vgp.VGP_kron(
                    data["Xtrain"],
                    data["Ytrain"],
                    np.arange(M),
                    a,
                    b,
                    kerns=prodkern(dim).kern_list,
                    likelihood=gpflow.likelihoods.Gaussian(),
                    use_two_krons=True,
                )
                m.likelihood.variance = noise_var

                # only optimize q(u)
                m.kerns.fixed = True
                m.likelihood.fixed = True

                start = time()
                m.optimize()
                marg_lik = m.compute_log_likelihood().squeeze()
                mean_log_pred = np.mean(m.predict_density(data["Xtest"], data["Ytest"]))
                t = time() - start

                results = results.append(
                    dict(
                        dim=dim,
                        rep=r,
                        marg_lik=marg_lik,
                        mean_log_pred=mean_log_pred,
                        time=t,
                        num_inducing=M,
                    ),
                    ignore_index=True,
                )

                # do this inside the loop so we can get partial results if something crashes
                results.to_csv("results/kron.csv")

##########################
# kron_opt
results = pd.DataFrame()

for dim in dimensions:
    a, b = -1.5 * np.ones(dim), 1.5 * np.ones(dim)
    k = prodkern(dim)
    for r in range(repeats):
        print("kron_opt replicate ", r, "/", repeats)
        data = np.load("data/data_dim{}_rep{}.npz".format(dim, r))
        for M in num_freqs:
            if (2 * M - 1) ** dim:
                m = VFF.vgp.VGP_kron(
                    data["Xtrain"],
                    data["Ytrain"],
                    np.arange(M),
                    a,
                    b,
                    kerns=k.kern_list,
                    likelihood=gpflow.likelihoods.Gaussian(),
                    use_two_krons=True,
                )
                m.likelihood.variance = noise_var
                # build kronecker GP model
                start = time()
                m.optimize()
                marg_lik = m.compute_log_likelihood().squeeze()
                mean_log_pred = np.mean(m.predict_density(data["Xtest"], data["Ytest"]))
                t = time() - start

                results = results.append(
                    dict(
                        dim=dim,
                        rep=r,
                        marg_lik=marg_lik,
                        mean_log_pred=mean_log_pred,
                        time=t,
                        num_inducing=M,
                    ),
                    ignore_index=True,
                )

                results.to_csv("results/kron_opt.csv")


##########################
# Sparse
results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        print("Sparse replicate ", r, "/", repeats)
        data = np.load("data/data_dim{}_rep{}.npz".format(dim, r))
        num_inducing = (2 * num_freqs - 1) ** dim
        for M in num_inducing:
            if M < 500:
                # build sparse GP model
                Z = KMeans(n_clusters=M).fit(data["Xtrain"]).cluster_centers_
                m = gpflow.sgpr.SGPR(data["Xtrain"], data["Ytrain"], Z=Z, kern=prodkern(dim))
                m.likelihood.variance = noise_var

                start = time()
                marg_lik = m.compute_log_likelihood().squeeze()
                mean_log_pred = np.mean(m.predict_density(data["Xtest"], data["Ytest"]))
                t = time() - start

                results = results.append(
                    dict(
                        dim=dim,
                        rep=r,
                        marg_lik=marg_lik,
                        mean_log_pred=mean_log_pred,
                        time=t,
                        num_inducing=M,
                    ),
                    ignore_index=True,
                )

                # do this inside the loop so we can get partial results if something crashes
                results.to_csv("results/sparse_kmeans.csv")


##########################
# Sparse GP opt
results = pd.DataFrame()

for dim in dimensions:
    for r in range(repeats):
        print("sparse opt replicate ", r, "/", repeats)
        data = np.load("data/data_dim{}_rep{}.npz".format(dim, r))
        num_inducing = (2 * num_freqs - 1) ** dim
        for M in num_inducing:
            if M < 500:
                # build sparse GP model
                Z = KMeans(n_clusters=M).fit(data["Xtrain"]).cluster_centers_
                m = gpflow.sgpr.SGPR(data["Xtrain"], data["Ytrain"], Z=Z, kern=prodkern(dim))
                m.likelihood.variance = noise_var

                # only optimize Z
                m.kern.fixed = True
                m.likelihood.fixed = True

                start = time()
                m.optimize()
                marg_lik = m.compute_log_likelihood().squeeze()
                mean_log_pred = np.mean(m.predict_density(data["Xtest"], data["Ytest"]))
                t = time() - start

                results = results.append(
                    dict(
                        dim=dim,
                        rep=r,
                        marg_lik=marg_lik,
                        mean_log_pred=mean_log_pred,
                        time=t,
                        num_inducing=M,
                    ),
                    ignore_index=True,
                )

                # do this inside the loop so we can get partial results if something crashes
                results.to_csv("results/sparse_opt.csv")


##########################
#
