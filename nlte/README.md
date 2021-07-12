
# AEcontext: Autoencoder for the context params (Stokes profiles)
# AEdataset: Autoencoder for the output params (Physical quantities)

# bayes_inversion: conditional normalizing flows for spectropolarimetric inversions
# bayes_inversion_encode_dataset: cflows using AE in output params
# forwardnet: ResNet to mimic the forward process (from physical params to Stokes)
# bayes_plot: produces all the plots

# nde_ae,nde_cvae,nde_nflow,nde_utils contain all the definitions of autoencoders and normalizing flows