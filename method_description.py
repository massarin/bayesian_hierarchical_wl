#!/usr/bin/env python
# coding: utf-8

# # The method in a nutshell
# 
# 
# We have a sample of lenses (e.g. massive galaxies) of which we wish to infer the distribution of dark matter halo masses (and maybe concentrations).
# We assume that:
# 1. Each lens dominates the weak lensing signal on background sources around it.
# 2. The center of each lens is known exactly.
# 3. The mass density profile of each lens can be described with a spherically symmetric Navarro Frenk & White (NFW) profile, plus a stellar component in the center.
# 
# Given these assumptions, each lens can be described with three parameters, for example, the stellar mass, halo mass and halo concentration: $M_*, M_{200}, c_{200}$.
# We refer to these collectively as the *individual object parameters*, $\psi$.
# 
# We assume that these parameters are drawn from a distribution describing the population of lenses, specified by a set of *hyper-parameters*, $\eta$. This distribution acts as a prior on the individual object parameters:
# $${\rm P}(\psi) = {\rm P}(\psi|\eta)$$
# In practice, the hyper-parameters $\eta$ will be things like the average halo mass, the halo mass-stellar mass correlation coefficient, the intrinsic scatter in halo mass, etc. (see subsection 3.2 of Sonnenfeld & Leauthaud 2018).
# 
# We want to infer the posterior probability distribution of the hyper-parametrs given the data, which consists of a set of shape measurements of lensed background sources and stellar mass measurements of the central galaxies of each halo. Using Bayes theorem, this is given by
# 
# $${\rm P}(\eta|d) \propto {\rm P}(\eta){\rm P}(d|\eta)$$
# 
# Making use of assumption 1. (a.k.a. the *isolated lens assumption*), we can write the likelihood ${\rm P}(d|\eta)$ as a product over each lens:
# 
# $${\rm P}(d|\eta) = \prod_i {\rm P}(d_i|\eta)$$
# 
# The likelihood of observing the data $d_i$ does not depend *directly* on the hyper-parameters $\eta$. The data is determined by the individual lens parameters $\psi$. To evaluate the likelihood term ${\rm P}(d_i|\eta)$ we need to marginalize over all possible values of $\psi$, given $\eta$:
# 
# $${\rm P}(d_i|\eta) = \int d\psi {\rm P}(d_i|\psi){\rm P}(\psi|\eta)$$
# 
# The term ${\rm P}(\psi|\eta)$ is the distribution of individual lens parameters given the model. We have some freedom in the choice of this distribution, as long as the model is a reasonable description of reality that captures the physical property that we're trying to measure. Some minimum requirements are: halo mass should scale with stellar mass and there should be a non-zero scatter in halo mass at fixed stellar mass. We can make our life easier by picking a distribution that's easy to work with, like a multivariate Gaussian.
# 
# The likelihood ${\rm P}(d_i|\psi)$ can be split in a term dependent on the observed stellar mass $M_{*,i}^{\mathrm{(obs)}}$ and a term dependent on the weak lensing shapes ${e_{j,i}\}$:
# 
# $${\rm P}(d_i|\psi) = {\rm P}(M_{*,i}^{\mathrm{(obs)}}|M_*){\rm P}(\{e_{j,i}\}|M_*,M_{200},c_{200}).$$
# 
# The second term is in general not an analytical function of the model parameters, which makes it complicated to calculate the 3D integral above (this needs to be computed for each lens at each step of an MCMC chain exploring the posterior PDF ${\rm P}(\eta|d)$).
# 
# The currently recommended approach for the calculation of the integral over $\psi$, is to adopt the following two-step trick:
# 1. Draw a sample of points $\{\psi^{(k)}\}$ from ${\rm P}(\psi|\eta)$
# 2. Approximate the integral as a weighted sum (Monte Carlo integration)
# 
# $$\int d\psi {\rm P}(d_i|\psi){\rm P}(\psi|\eta) \approx \frac{1}{N} \sum_k {\rm P}(d_i|\psi^{(k)})$$
# 
# And to speed up computations, it is recommended to pre-compute the weak lensing likelihood on a grid of values of $\psi$.
# This is a slightly different approach than the one advertised in the original Sonnenfeld & Leauthaud (2018) paper, but it has been found to provide a more accurate inference of the intrinsic scatter.
# 
# We can then run a Markov Chain Monte Carlo to sample the posterior probability distribution ${\rm P}(\eta|d)$.
