# Bayesian Stokes inversion with Normalizing flows

This repository contains the code to perform Bayesian inference of the solar atmosphere using normalizing flows, producing the posterior distribution of solar physical parameters from observed Stokes parameters ([https://arxiv.org/abs/2108.07089](https://arxiv.org/abs/2108.07089)).


> A simple tutorial on the application of normalizing flows in a linear regression problem and comparison with an MCMC method has been added in the following [notebook](https://github.com/cdiazbas/bayesflows/blob/main/nflows_example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdiazbas/bayesflows/blob/main/nflows_example.ipynb)



<!-- ![example](milne/milne.png?raw=true "") -->
![example](nlte/nlte.png?raw=true "")

**Figure 1** — Atmospheric stratification inferred by the normalizing flow for two examples. In each column, the orange solution is inferred only using the Fe I line while the brown solution also uses the Ca II profile. The lowest row shows the original intensity values together with the synthetic calculation from the maximum posterior solution.

## Abstract
Stokes inversion techniques are very powerful methods for obtaining information on the thermodynamic and magnetic properties of solar and stellar atmospheres. In recent years, very sophisticated inversion codes have been developed that are now routinely applied to spectro-polarimetric observations. Most of these inversion codes are designed for finding the optimum solution of the nonlinear inverse problem. However, to obtain the location of potentially multimodal cases (ambiguities), the degeneracies, and the uncertainties of each parameter inferred from the inversions, algorithms such as Markov chain Monte Carlo (MCMC) requires to evaluate the likelihood of the model thousand of times and are computationally costly. Variational methods are a quick alternative to Monte Carlo methods by approximating the posterior distribution by a parametrized distribution. In this study, we introduce a highly flexible variational inference method to perform fast Bayesian inference, known as normalizing flows. Normalizing flows are a set of invertible, differentiable, and parametric transformations that convert a simple distribution into an approximation of any other complex distribution. If the transformations are conditioned on observations, the normalizing flows can be trained to return Bayesian posterior probability estimates for any observation. We illustrate the ability of the method using a simple Milne-Eddington model and a complex non-LTE inversion. However, the method is extremely general and other more complex forward models can be applied. The training procedure need only be performed once for a given prior parameter space and the resulting network can then generate samples describing the posterior distribution several orders of magnitude faster than existing techniques.