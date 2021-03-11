# VFF for GPflow 2

This is an updated version of the Variational Fourier Features implementation, forked from https://github.com/jameshensman/VFF (which was originally written to accompany the paper ["Variational Fourier Features for Gaussian Processes" by James Hensman, Nicolas Durrande and Arno Solin](http://www.jmlr.org/papers/v18/16-579.html)).

Compared to the base repo, this version of the code has been ported to TensorFlow 2.x and GPflow 2. Some bugs and issues have been fixed in the process, and more tests of the code have been added. This is still work in progress.
So far, the following modules have been ported:
- [ ] [gpmc.py](https://github.com/st--/VFF/blob/gpflow2/VFF/gpmc.py)
- [ ] [gpr.py](https://github.com/st--/VFF/blob/gpflow2/VFF/gpr.py)
- [X] [kronecker_ops.py](https://github.com/st--/VFF/blob/gpflow2/VFF/kronecker_ops.py)
- [X] [matrix_structures.py](https://github.com/st--/VFF/blob/gpflow2/VFF/matrix_structures.py)
- [ ] [psi_statistics.py](https://github.com/st--/VFF/blob/gpflow2/VFF/psi_statistics.py)
- [ ] [sfgpmc_kronecker.py](https://github.com/st--/VFF/blob/gpflow2/VFF/sfgpmc_kronecker.py)
- [X] [spectral_covariance.py](https://github.com/st--/VFF/blob/gpflow2/VFF/spectral_covariance.py)
- [ ] [ssgp.py](https://github.com/st--/VFF/blob/gpflow2/VFF/ssgp.py)
- [X] [vgp.py](https://github.com/st--/VFF/blob/gpflow2/VFF/vgp.py)


### Install
VFF relies heavily on [GPflow](github.com/GPflow/GPflow). After installing GPflow, clone this repo and add the VFF directory to your PYTHONPATH. There are some examples in the `experiments` directory that replicate figures from the manuscript. 

Questions and comments are welcome via github issues on this repo.



