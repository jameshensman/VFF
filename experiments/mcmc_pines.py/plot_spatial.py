import numpy as np
from matplotlib import pylab as plt

X = np.load('pines.np')
fig = plt.figure()

ax1 = fig.add_subplot(141,aspect='equal')
ax2 = fig.add_subplot(142,aspect='equal')
ax3 = fig.add_subplot(143,aspect='equal')
ax4 = fig.add_subplot(144,aspect='equal')

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])

xticklabels = ax1.get_xticklabels()+ ax2.get_xticklabels() + ax3.get_xticklabels()

ax1.scatter( X[:,0], X[:,1] )
ax1.set_xlim([0., 1.])
ax1.set_ylim([0., 1.])

gold_standard_intensities_32 = np.loadtxt( 'gold_standard_intensities_32_grid_comma.np' )

ax2.imshow( gold_standard_intensities_32, interpolation='none',vmin=0.,vmax=12. )

variational_intensities_32 = np.loadtxt( '225_inducing_point_intensities_32_comma', delimiter=',' )

ax3.imshow( variational_intensities_32, interpolation='none', vmin=0., vmax=12. )

variational_intensities_64 =  np.loadtxt( '225_inducing_point_intensities_64_comma', delimiter=',' )

im = ax4.imshow( variational_intensities_64, interpolation='none', vmin=0., vmax=12. )

fig.subplots_adjust(right=0.875)
cbar_ax = fig.add_axes([0.9, 0.385, 0.04, 0.225])

fig.colorbar(im, cax=cbar_ax)

#from matplotlib2tikz import save as save_tikz

#save_tikz('pines.tikz',figurewidth='\\figurewidth', figureheight = '\\figureheight')

plt.show()

