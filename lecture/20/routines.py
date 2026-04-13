###################################################################################################
#
# AST 5011 - Lecture 20: AGN & Black Holes
#
# Helper routines for AGN physics calculations
#
###################################################################################################

import numpy as np

from colossus.utils import constants
from colossus.cosmology import cosmology

###################################################################################################
# Physical constants for AGN
###################################################################################################

sigma_T = 6.65e-25     # Thomson cross section (cm^2)
L_sun = 3.83e33         # Solar luminosity (erg/s)

# Default radiative efficiency
eps_rad_default = 0.1

###################################################################################################
# AGN functions
###################################################################################################

def eddingtonLuminosity(M_bh):
    """Eddington luminosity (erg/s) for BH mass M_bh (Msun)."""
    return 4.0 * np.pi * constants.C * constants.G_CGS * constants.M_PROTON / sigma_T * M_bh * constants.MSUN

def eddingtonAccretionRate(M_bh, eps_rad=0.1):
    """Eddington accretion rate (Msun/yr) for BH mass M_bh (Msun)."""
    L_edd = eddingtonLuminosity(M_bh)
    return L_edd / (eps_rad * constants.C**2) / constants.MSUN * constants.YEAR

def salpeterTime(eps_rad=0.1):
    """Salpeter (e-folding) time in years."""
    return sigma_T * constants.C / (4.0 * np.pi * constants.G_CGS * constants.M_PROTON) / constants.YEAR * eps_rad

def bhGrowth(M_seed, t_grow, eps_rad=0.1):
    """BH mass after growing at Eddington rate for t_grow years from M_seed (Msun)."""
    t_salp = salpeterTime(eps_rad)
    return M_seed * np.exp(t_grow / t_salp)

def accretionLuminosity(Mdot, eps_rad=0.1):
    """Accretion luminosity (erg/s) for accretion rate Mdot (Msun/yr)."""
    return eps_rad * Mdot * constants.MSUN / constants.YEAR * constants.C**2

###################################################################################################
# M_BH - sigma relation (Kormendy & Ho 2013)
###################################################################################################

def mbhSigmaRelation(sigma):
    """M_BH from the M-sigma relation (Kormendy & Ho 2013). sigma in km/s, returns M_BH in Msun."""
    return 10**(8.49 + 4.38 * np.log10(sigma / 200.0))

def mbhMbulgeRelation(M_bulge):
    """M_BH from the M_BH-M_bulge relation (Kormendy & Ho 2013). M_bulge in Msun."""
    return 10**(8.46 + 1.05 * np.log10(M_bulge / 1e11))

###################################################################################################
# Cooling flow and jet power functions
###################################################################################################

def coolingLuminosity(T_keV, n_e, r_cool_kpc):
    """
    Approximate X-ray cooling luminosity of a cluster core (erg/s).

    Uses bremsstrahlung cooling: Lambda ~ 1.4e-27 * T^{1/2} * n_e^2 * V
    T_keV: gas temperature in keV
    n_e: electron density in cm^{-3}
    r_cool_kpc: cooling radius in kpc
    """
    T_K = T_keV * 1.16e7  # keV to K
    r_cm = r_cool_kpc * 3.086e21  # kpc to cm
    V = (4.0 / 3.0) * np.pi * r_cm**3
    Lambda_brem = 1.4e-27 * np.sqrt(T_K)  # erg cm^3 / s
    return Lambda_brem * n_e**2 * V

def coolingTime(T_keV, n_e):
    """
    Cooling time (yr) for gas at temperature T_keV and density n_e (cm^{-3}).

    t_cool = (3/2) n k_B T / (n_e^2 Lambda)
    """
    T_K = T_keV * 1.16e7
    k_B = 1.381e-16  # erg/K
    Lambda_brem = 1.4e-27 * np.sqrt(T_K)
    t_cool_s = 1.5 * n_e * k_B * T_K / (n_e**2 * Lambda_brem)
    return t_cool_s / constants.YEAR

def jetPower(M_bh, mdot_frac, eta_jet=0.1):
    """
    Mechanical jet power (erg/s) for a BH accreting at mdot_frac * Mdot_Edd.

    eta_jet: fraction of accretion power that goes into jet kinetic energy
    """
    L_edd = eddingtonLuminosity(M_bh)
    return eta_jet * mdot_frac * L_edd

def massDepositionRate(L_cool, T_keV):
    """
    Classical cooling flow mass deposition rate (Msun/yr).

    Mdot_cool = 2 * L_cool * mu * m_p / (5 * k_B * T)
    """
    T_K = T_keV * 1.16e7
    k_B = 1.381e-16
    mu = 0.6  # mean molecular weight
    Mdot_cgs = 2.0 * L_cool * mu * constants.M_PROTON / (5.0 * k_B * T_K)
    return Mdot_cgs / constants.MSUN * constants.YEAR

def eddingtonRatio(L_agn, M_bh):
    """Eddington ratio lambda = L_AGN / L_Edd."""
    return L_agn / eddingtonLuminosity(M_bh)

###################################################################################################
# Cosmology setup
###################################################################################################

cosmo = cosmology.setCosmology('planck18')

###################################################################################################
