"""Exact fixed-template Gaussian detection and binary denoising bounds."""
import numpy as np
from scipy.special import ndtr,ndtri,expit
from scipy.stats import norm


def detection_power(mu,alpha=.01):
    return ndtr(np.asarray(mu)-norm.isf(alpha))


def minimum_sigma(energy,repeats=16,clip=1.,alpha=.01,beta=.05):
    if not 0<alpha<beta<1:raise ValueError('Require 0 < alpha < beta < 1')
    return np.sqrt(np.asarray(energy)*repeats)/(clip*(norm.isf(alpha)+ndtri(beta)))


def posterior_probability(score,mu):
    return expit(np.asarray(mu)*np.asarray(score)-np.asarray(mu)**2/2)


def binary_mse_lower_bound(mu):
    # Posterior variance p(1-p) >= min(p,1-p)/2, integrated over observations.
    # The optimal equal-prior Gaussian classification error is Phi(-mu/2).
    return .5*ndtr(-np.asarray(mu)/2)
