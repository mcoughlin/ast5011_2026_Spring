###################################################################################################
#
# AST 5011 - Lecture 18: Stars & Stellar Populations
#
# Helper routines adapted from Benedikt Diemer's ASTR 620 course
#
###################################################################################################

import numpy as np
import os

from colossus.cosmology import cosmology

###################################################################################################
# Constants
###################################################################################################

data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# Solar metallicity
Z_SOLAR = 0.0134

# Solar absolute magnitudes (Sparke & Gallagher Table 1.3)
solar_mag = {'u': 6.55, 'g': 5.12, 'r': 4.68, 'i': 4.57, 'z': 4.60, 'bol': 4.74}

# MW values from Licquia & Newman 2016
cosmo = cosmology.setCosmology('planck18')
Mstar_MW = 5.71e10
SFR_MW = 1.65
Mr_MW = -20.97 + 5.0 * np.log10(cosmo.h)
gr_MW = 0.678

###################################################################################################
# Initial Mass Functions
###################################################################################################

def imfSalpeter(m):
    """Salpeter (1955) IMF: dn/dlog10(m) ~ m^-1.35."""
    return m**(-1.35)

def imfKroupa(m):
    """Kroupa (2001) broken power-law IMF."""
    m = np.atleast_1d(np.array(m, dtype=float))
    imf = np.zeros_like(m)
    mask1 = (m < 0.5)
    imf[mask1] = 2.0 * m[mask1]**(-0.3)
    mask2 = (m >= 1.0)
    imf[mask2] = m[mask2]**(-1.7)
    mask3 = ~mask1 & ~mask2
    imf[mask3] = m[mask3]**(-1.3)
    return imf

def imfChabrier(m):
    """Chabrier (2003) IMF: lognormal below 1 Msun, power-law above."""
    m = np.atleast_1d(np.array(m, dtype=float))
    A = 0.158
    mc = 0.079
    sig = 0.69
    imf = A * np.exp(-0.5 * (np.log10(m / mc))**2 / sig**2)
    mask = (m > 1.0)
    imf[mask] = 4.43e-2 * m[mask]**(-1.3)
    return imf

###################################################################################################
# Star Formation Histories
###################################################################################################

def sfrExponential(t, tau):
    """Exponential decay SFH: psi ~ exp(-t/tau)."""
    return np.exp(-t / tau)

def sfrDelayedTau(t, tau):
    """Delayed-tau SFH: psi ~ (t/tau) * exp(-t/tau)."""
    return (t / tau) * np.exp(-t / tau)

def sfrLognormal(t, t0, tau):
    """Lognormal SFH."""
    A = 1.0 / (np.log(tau) * np.sqrt(2.0 * np.pi) * t)
    return A * np.exp(-0.5 * (np.log(t) - np.log(t0) - np.log(tau)**2)**2 / np.log(tau)**2)

###################################################################################################
# Mass-to-light ratio
###################################################################################################

def mlZibetti(gr):
    """M*/L_r from g-r color (Zibetti et al. 2009)."""
    return 10**(-0.840 + 1.654 * gr)

###################################################################################################
# SDSS data loader
###################################################################################################

def loadSdssSubset():
    """Load pre-extracted SDSS galaxy subset."""
    d = np.load(data_dir + 'sdss_subset.npz', allow_pickle=True)
    return d['data']

###################################################################################################
