###################################################################################################
#
# AST 5011 - Lecture 17: The Galaxy-Halo Connection & Gas Accretion
#
# Helper routines adapted from Benedikt Diemer's ASTR 620 course
#
###################################################################################################

import numpy as np
import os
import scipy.integrate
import h5py
from scipy.interpolate import RegularGridInterpolator

from colossus.utils import constants
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.halo import mass_so

###################################################################################################

# Path to N-body simulation data (Erebos suite)
nbody_data_dir = '/Users/mcoughlin/Code/Teaching/AST5011/ast5011_2026_Spring_workspace/galaxies_course/data/'

# Path to cooling table
data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'

# Mean molecular weights
mu_ion = 0.59    # fully ionized primordial gas
mu_h = 1.4       # hydrogen nuclei (including He weight)

# Solar metallicity (Asplund+2009)
Z_SOLAR = 0.0134

###################################################################################################
# Schechter function
###################################################################################################

def schechterFunctionMass(M, phi_star, M_knee, alpha, units='dn_dln'):
    """
    Schechter mass function.

    Parameters
    ----------
    M : ndarray
        Mass array (same units as M_knee).
    phi_star : float
        Normalization (number density at the knee).
    M_knee : float
        Characteristic mass.
    alpha : float
        Faint-end slope.
    units : str
        'dn_dln' for dn/d(ln M), 'dn_dlog10' for dn/d(log10 M), 'dn' for dn/dM.
    """
    x = M / M_knee
    phi = phi_star * x**(alpha + 1) * np.exp(-x)
    if units == 'dn_dln':
        return phi
    elif units == 'dn_dlog10':
        return phi * np.log(10)
    elif units == 'dn':
        return phi / M
    else:
        raise ValueError('Unknown units: %s' % units)

###################################################################################################
# Stellar-halo mass relations from observations
###################################################################################################

def mstarBehroozi13(Mhalo, z=0.0):
    """
    Fitting function from Behroozi et al. 2013.
    Mhalo in Msun, returns Mstar in Msun.
    """
    def bfz(x, z):
        alpha_0, alpha_a = -1.412, 0.731
        delta_0, delta_a, delta_z = 3.508, 2.608, -0.043
        gamma_0, gamma_a, gamma_z = 0.316, 1.319, 0.279
        a = 1.0 / (1.0 + z)
        nu = np.exp(-4.0 * a**2)
        alpha = alpha_0 + nu * (alpha_a * (a - 1.0))
        delta = delta_0 + nu * (delta_a * (a - 1.0) + delta_z * z)
        gamma = gamma_0 + nu * (gamma_a * (a - 1.0) + gamma_z * z)
        return -np.log10(10**(alpha * x) + 1.0) + delta * (np.log10(1.0 + np.exp(x)))**gamma / (1.0 + np.exp(10**(-x)))

    M1_0, M1_a, M1_z = 11.514, -1.793, -0.251
    eps0, eps_a, eps_z, eps_a2 = -1.777, -0.006, 0.0, -0.119
    a = 1.0 / (1.0 + z)
    nu = np.exp(-4.0 * a**2)
    log_M1 = M1_0 + nu * (M1_a * (a - 1.0) + z * M1_z)
    log_eps = eps0 + nu * (eps_a * (a - 1.0) + eps_z * z) + eps_a2 * (a - 1.0)
    log_Mstar = log_eps + log_M1 + bfz(np.log10(Mhalo) - log_M1, z) - bfz(0.0, z)
    return 10**log_Mstar

def shmrBehroozi13():
    """Return (Mvir, Mstar/Mvir) from Behroozi+2013 at z=0."""
    Mvir = 10**np.linspace(9.0, 16.0, 200)
    mstar = mstarBehroozi13(Mvir, 0.0)
    mask = (mstar >= 10**7.25)
    return Mvir[mask], mstar[mask] / Mvir[mask]

def mstarKravtsov18(Mvir):
    """Fitting function from Kravtsov et al. 2018 at z=0. Mvir in Msun."""
    log_M1, log_eps, alpha, delta, gamma = 11.43, -1.663, -1.750, 4.290, 0.595
    def f(x):
        return -np.log10(10**(alpha * x) + 1.0) + delta * (np.log10(1.0 + np.exp(x)))**gamma / (1.0 + np.exp(10**(-x)))
    epsilon = 10**log_eps
    M1 = 10**log_M1
    log_Mstar = np.log10(epsilon * M1) + f(np.log10(Mvir / M1)) - f(0.0)
    return 10**log_Mstar

def shmrKravtsov18():
    """Return (Mvir, Mstar/Mvir) from Kravtsov+2018 at z=0."""
    Mvir = 10**np.linspace(10.0, 15.2, 200)
    mstar = mstarKravtsov18(Mvir)
    return Mvir, mstar / Mvir

###################################################################################################
# Cumulative mass function helpers
###################################################################################################

def cumulativeStellarMF(M_star_arr, phi_star, M_knee, alpha):
    """
    Compute cumulative stellar mass function n(>M_star) by integrating Schechter.
    """
    n_cumul = np.zeros_like(M_star_arr)
    for i, M_low in enumerate(M_star_arr):
        integrand = lambda logM: schechterFunctionMass(10**logM, phi_star, M_knee, alpha, units='dn_dlog10')
        val, _ = scipy.integrate.quad(integrand, np.log10(M_low), 14.0, epsrel=1e-4)
        n_cumul[i] = val
    return n_cumul

def cumulativeHaloMF(M_halo_arr, z=0.0, mdef='200m'):
    """
    Compute cumulative halo mass function n(>M_halo) using Colossus Tinker08.
    """
    n_cumul = np.zeros_like(M_halo_arr)
    for i, M_low in enumerate(M_halo_arr):
        integrand = lambda logM: mass_function.massFunction(10**logM, z, q_in='M',
                                                             q_out='dndlnM', model='tinker08',
                                                             mdef=mdef) / np.log(10.0)
        val, _ = scipy.integrate.quad(integrand, np.log10(M_low), 17.0, epsrel=1e-4)
        n_cumul[i] = val
    return n_cumul

###################################################################################################
# Cooling functions (Ploeckinger & Schaye 2020)
###################################################################################################

def coolingInterpolator(Z):
    """
    Build a 2D interpolator (T, n) for the net cooling rate at metallicity Z
    (in solar units) using the Ploeckinger & Schaye (2020) tables at z=0.

    Returns (Tmin, Tmax, nmin, nmax, interp).
    """
    data_path = data_dir + 'ploeckinger_schaye_20_z0.hdf5'

    f = h5py.File(data_path, 'r')
    bins_Z = np.array(f['bins_metallicity'])
    bins_T = np.array(f['bins_temperature'])
    bins_n = np.array(f['bins_density'])
    cooling_primordial = np.array(f['cooling_primordial'])
    cooling_metal = np.array(f['cooling_metal'])
    heating_primordial = np.array(f['heating_primordial'])
    heating_metal = np.array(f['heating_metal'])
    f.close()

    # Total = Primordial + Metal, and net cooling = Lambda - Gamma
    cool_rate = 10**cooling_primordial + 10**cooling_metal
    heat_rate = 10**heating_primordial + 10**heating_metal
    total_rate = cool_rate - heat_rate
    total_rate[total_rate <= 1E-40] = 1E-40
    total_rate = np.log10(total_rate)
    rgi1 = RegularGridInterpolator((bins_T, bins_Z, bins_n), total_rate, method='linear')

    # Fix metallicity and create 2D interpolator
    Z_use = np.log10(Z * Z_SOLAR / 0.0134)
    grid_T, grid_n = np.meshgrid(bins_T, bins_n, indexing='ij')
    grid_Z = np.ones_like(grid_T) * Z_use
    grid = np.stack((grid_T.flatten(), grid_Z.flatten(), grid_n.flatten())).T

    grid_fixed_Z = rgi1(grid)
    grid_fixed_Z = grid_fixed_Z.reshape(len(bins_T), len(bins_n))
    interp = RegularGridInterpolator((bins_T, bins_n), grid_fixed_Z, method='linear')

    Tmin = 10**bins_T[0]
    Tmax = 10**bins_T[-1]
    nmin = 10**bins_n[0]
    nmax = 10**bins_n[-1]

    return Tmin, Tmax, nmin, nmax, interp

###################################################################################################

def coolingRate(Z, T, n):
    """
    Net cooling rate Lambda (erg/s/cm^3) at metallicity Z (solar), temperature T (K),
    and number density n (cm^-3). Note: returns Lambda, NOT Lambda/n^2.
    """
    _, _, _, _, interp = coolingInterpolator(Z)

    T_array = np.atleast_1d(np.array(T, dtype=float))
    n_array = np.atleast_1d(np.array(n, dtype=float))
    is_scalar = (np.ndim(T) == 0 and np.ndim(n) == 0)

    if len(T_array) == 1 and len(n_array) > 1:
        T_array = np.ones_like(n_array) * T_array[0]
    elif len(n_array) == 1 and len(T_array) > 1:
        n_array = np.ones_like(T_array) * n_array[0]

    grid = np.stack((np.log10(T_array), np.log10(n_array))).T
    Lambda = 10**interp(grid) * n_array**2

    if is_scalar:
        return Lambda[0]
    return Lambda

###################################################################################################
# Virial quantities
###################################################################################################

def rhoVir(z):
    """Virial density (total, including DM) in g/cm^3."""
    cosmo = cosmology.getCurrent()
    return mass_so.densityThreshold(z, 'vir') * constants.MSUN * cosmo.h**2 / constants.KPC**3

def TvirFromMvir(Mvir, z, mu=mu_ion):
    """Virial temperature (K) for halo mass Mvir (Msun)."""
    cosmo = cosmology.getCurrent()
    Mvir_h = Mvir * cosmo.h
    Rvir_h = mass_so.M_to_R(Mvir_h, z, 'vir')
    Vvir = np.sqrt(constants.G * Mvir_h / Rvir_h) * 1E5  # km/s -> cm/s
    T = mu * constants.M_PROTON / constants.KB / 3 * Vvir**2
    return T

def MvirFromTvir(Tvir, z, mu=mu_ion):
    """Halo mass (Msun) corresponding to a given virial temperature."""
    Mvir = (3 * constants.KB * Tvir / (mu * constants.M_PROTON * constants.G_CGS))**(3.0 / 2.0) \
        * np.sqrt(3 / (4 * np.pi * rhoVir(z))) / constants.MSUN
    return Mvir

###################################################################################################
# Timescales
###################################################################################################

def n_tcool(Z, T, n):
    """Cooling time times n (s * cm^-3). Lambda is the total cooling rate."""
    Lambda_T = coolingRate(Z, T, n)
    tcool = 3.0 * constants.KB * T / (2.0 * Lambda_T) * n**2
    return tcool

def tcoolFromMvir(Mvir, z, Z):
    """Cooling time (s) for halo of mass Mvir at virial density, metallicity Z."""
    cosmo = cosmology.getCurrent()
    Tvir = TvirFromMvir(Mvir, z)
    rho_vir = rhoVir(z)
    fb = cosmo.Ob(0.0) / cosmo.Om(0.0)
    n_vir = rho_vir * fb / (mu_h * constants.M_PROTON)
    tcool = n_tcool(Z, Tvir, n_vir) / n_vir
    return tcool

def tFreeFall(rho):
    """Free-fall time (s) from total density rho (g/cm^3)."""
    return np.sqrt(3.0 * np.pi / (32.0 * constants.G_CGS * rho))

def tFreeFallVir(z):
    """Free-fall time (s) at the virial density at redshift z."""
    return tFreeFall(rhoVir(z))

###################################################################################################
# Cosmology setup
###################################################################################################

cosmo = cosmology.setCosmology('planck18')

###################################################################################################
