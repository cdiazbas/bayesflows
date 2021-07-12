# Complex case: NLTE with stratification.

- The generation of the database requires the installation of  `STiC` - https://github.com/jaimedelacruz/stic, `nflows` - https://github.com/bayesiains/nflows, and `corner` - https://github.com/dfm/corner.py.

## Training and results

- `AEcontext.py`: Autoencoder for the context params (Stokes profiles)
- `AEdataset.py`: Autoencoder for the output params (Physical quantities)

- `bayes_inversion.py`: conditional normalizing flows for spectropolarimetric inversions
- `bayes_inversion_encode_dataset.py`: cflows using AE in output params
- `forwardnet.py`: ResNet to mimic the forward process (from physical params to Stokes)
- `bayes_plot.py`: produces all the plots

- `nde_ae.py`, `nde_cvae.py`, `nde_nflow.py`, s`nde_utils.py` contain all the definitions of autoencoders and normalizing flows