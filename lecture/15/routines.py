###################################################################################################
#
# AST 5011 - Lecture 16: Density Fluctuations in the Early Universe
#
# Helper routines adapted from Benedikt Diemer's ASTR 620 course
#
###################################################################################################

import numpy as np
from colossus.cosmology import cosmology

###################################################################################################

def filterFourierSpace(kR, filt='tophat'):
    """Smoothing filter in Fourier space (tophat or gaussian)."""
    if filt == 'tophat':
        return 3.0 * (np.sin(kR) - kR * np.cos(kR)) / (kR)**3
    elif filt == 'gaussian':
        return np.exp(-0.5 * kR**2)
    else:
        raise ValueError('Unknown filter: %s' % filt)

###################################################################################################

def gaussianRandomField(Lbox, N, z_ini, Pk_z0_func, **kwargs):
    """
    Generate a 3D Gaussian random field from an input power spectrum.

    Parameters
    ----------
    Lbox : float
        Box size in Mpc/h.
    N : int
        Number of grid cells per dimension.
    z_ini : float
        Initial redshift (used to scale the growth factor).
    Pk_z0_func : callable
        Function that returns P(k) at z=0 given wavenumber k in h/Mpc.

    Returns
    -------
    k : ndarray
        3D array of wavenumber magnitudes.
    delta : ndarray
        3D real-space density field.
    delta_k : ndarray
        3D Fourier-space density field.
    D : float
        Linear growth factor at z_ini.
    """
    cosmo = cosmology.getCurrent()

    N3 = N * N * N
    dx = Lbox / N
    dxi = 1.0 / dx

    # Generate white noise
    delta = np.reshape(np.random.normal(0.0, 1.0, N3), (N, N, N))
    delta_k = np.fft.fftn(delta)

    # Compute |k| for each grid point
    k1d = np.fft.fftfreq(N) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d)
    k = dxi * np.sqrt(kx**2 + ky**2 + kz**2)
    k[0, 0, 0] = 2.0 * np.pi / Lbox

    # Weight by sqrt(P(k)) * D(z)
    D = cosmo.growthFactor(z_ini)
    Pk = Pk_z0_func(k, **kwargs) * D**2
    delta_k *= np.sqrt(Pk)

    # Zero the DC mode
    delta_k[0, 0, 0] = 0.0

    # Transform back to real space
    delta = np.real(np.fft.ifftn(delta_k))

    return k, delta, delta_k, D

###################################################################################################

def powerSpectrumWarmDM(k, wdm_mass=0.4):
    """
    Approximate warm dark matter power spectrum (Lovell et al. 2014, Eq. 3).
    Suppresses small-scale power relative to CDM.
    """
    from routines import cosmo
    P = cosmo.matterPowerSpectrum(k)
    wdm_nu = 1.0
    alpha = 0.05 * wdm_mass**-1.15 * (cosmo.Om0 / 0.4)**0.15 * \
            (cosmo.h / 0.65)**1.3 * (1.5 / 1.5)**-0.29
    T = (1.0 + (alpha * k)**(2.0 * wdm_nu))**-(5.0 / wdm_nu)
    P *= T**2
    return P

###################################################################################################
# Cosmology setup
###################################################################################################

cosmo = cosmology.setCosmology('planck18')

###################################################################################################
