import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
plt.ion()
plt.close('all')

from config import *

def plot_all(dim):
    d_full = pd.read_csv('results/full.csv')
    d_sparse = pd.read_csv('results/sparse.csv')
    d_vff = pd.read_csv('results/vff.csv')
    bullet = ['o', 'X', 'D', 'P', 'H', 'h', '*', 'p']
    col = ['C0', 'C1']
    names = ['VFF', 'sparse']
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    for i, d in enumerate([d_vff, d_sparse]):
        df = pd.merge(d[d.dim==dim], d_full[d_full.dim==dim], on=['rep'])
        for j, num_basis in enumerate(num_Basis[dim-1]):
            ind_basis = df.num_inducing == num_basis ** dim
            ax.plot(df.time[ind_basis],-df.marg_lik_x[ind_basis] + df.marg_lik_y[ind_basis], bullet[j], color=col[i])  # KL
            #ax.plot(df.time[ind_basis],-df.mean_log_pred_x[ind_basis] + df.mean_log_pred_y[ind_basis], bullet[j], color=col[i]) # log pred
        # ax.set_ylim((-0.02,1.6))
        ax.semilogx()
        ax.set_title('dimension {}'.format(dim))
    # make legend
    points_vff = ax.plot(np.inf, np.inf, 's', color='C0', label='vff')
    points_spa = ax.plot(np.inf, np.inf, 's', color='C1', label='sparse')
    plt.legend()
    for j, num_basis in enumerate(num_Basis[dim-1]):
        ax.plot(np.inf ,np.inf , bullet[j], color='k', label='$M = {}^{}$'.format(num_basis,dim))
    ax.set_xlabel('time (s)')
    plt.legend()

def plot_all_all():
    d_full = pd.read_csv('results/full.csv')
    d_sparse = pd.read_csv('results/sparse.csv')
    d_vff = pd.read_csv('results/vff.csv')
    bullet = ['o', 'X', 'D', 'P', 'H', 'h', '*', 'p']
    col = ['C0', 'C1']
    names = ['VFF', 'sparse']
    fig = plt.figure(figsize=(15,30))
    k = 0
    for dim in dimensions:
        for a in A:
            k += 1
            ax = plt.subplot(4, A.shape[0], k)
            dv = d_vff.loc[np.isclose(d_vff['a'], a)] 
            for i, d in enumerate([dv, d_sparse]):
                df = pd.merge(d[d.dim==dim], d_full[d_full.dim==dim], on=['rep'])
                for j, num_basis in enumerate(num_Basis[dim-1]):
                    ind_basis = df.num_inducing == num_basis ** dim
                    ax.plot(df.time[ind_basis],-df.marg_lik_x[ind_basis] + df.marg_lik_y[ind_basis], bullet[j], color=col[i])  # KL
                    #ax.plot(df.time[ind_basis],-df.mean_log_pred_x[ind_basis] + df.mean_log_pred_y[ind_basis], bullet[j], color=col[i]) # log pred
                # ax.set_ylim((-0.02,1.6))
                ax.semilogx()
                ax.set_title('dimension {}, a = {}'.format(dim, np.round(a,1)))
            # make legend
            points_vff = ax.plot(np.inf, np.inf, 's', color='C0', label='vff')
            points_spa = ax.plot(np.inf, np.inf, 's', color='C1', label='sparse')
            plt.legend()
            for j, num_basis in enumerate(num_Basis[dim-1]):
                ax.plot(np.inf ,np.inf , bullet[j], color='k', label='$M = {}^{}$'.format(num_basis,dim))
            ax.set_xlabel('time (s)')
            plt.legend()
    
# for d in range(1,5):
#     plot_all(d)

plot_all_all()

input("Press Enter to continue...")
# plot_pred(pd.read_csv('results/sparse_opt.csv', index_col=0))
# plot_pred(pd.read_csv('results/sparse_kmeans.csv', index_col=0))
# plot_pred(pd.read_csv('results/kron_opt.csv', index_col=0))
# plot_pred(pd.read_csv('results/kron.csv', index_col=0))

# plot_KL(pd.read_csv('sparse_opt.csv', index_col=0))
# plot_KL(pd.read_csv('sparse_kmeans.csv', index_col=0))
# plot_KL(pd.read_csv('kron_opt.csv', index_col=0))
