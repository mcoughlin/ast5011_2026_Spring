###################################################################################################
#
# AST 5011 - Lecture 19: Galaxy-Halo Coevolution & Feedback
#
# Semi-analytic galaxy evolution model adapted from Benedikt Diemer's ASTR 620 course
#
###################################################################################################

import numpy as np
import scipy.integrate

from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import mass_function

###################################################################################################
# Component indices
###################################################################################################

IH = 0   # Halo mass
IG = 1   # Gas mass
IS = 2   # Stellar mass
IZ = 3   # Metal mass

###################################################################################################
# GalaxyModel class
###################################################################################################

class GalaxyModel():
    """
    Semi-analytic model for galaxy evolution.

    Tracks the co-evolution of dark matter halo mass, gas mass, stellar mass,
    and metal mass via coupled ODEs. Includes models for:
    - Halo mass accretion (Neistein+2008)
    - Reionization suppression (Okamoto+2008)
    - Cooling cutoff at high masses
    - Star formation (gas depletion time)
    - Stellar feedback / winds (Muratov+2015)
    """

    def __init__(self, z_ini, Mh_ini,
                 # Metallicity
                 Z_in=2E-5, yZ=0.069,
                 # MAH model
                 model_mah='neistein08',
                 # Reionization model
                 model_reion='okamoto08', z_reion=9.0,
                 M_c1_early=1E7, M_c1_0=2E10, alpha1=2.0,
                 # Cooling model
                 model_cooling='cutoff', M_c2_0=1E12, alpha2=1.0,
                 # SFR model
                 model_sfr='depl_time', t_depl=2.0, R_loss=0.46,
                 # Wind/feedback model
                 model_wind='muratov15_star', eta_fixed=1.0,
                 eta_min=2.0, eta_max=300.0):

        self.cosmo = cosmology.getCurrent()
        self.fb = self.cosmo.Ob(0.0) / self.cosmo.Om(0.0)

        # Initial conditions
        self.z_ini = z_ini
        self.n_comp = 4
        self.Mx_ini = np.zeros(self.n_comp)
        self.Mx_ini[IH] = Mh_ini
        self.Mx_ini[IG] = Mh_ini * self.fb
        self.Mx_ini[IZ] = self.Mx_ini[IG] * Z_in

        # Parameters
        self.Z_in = Z_in
        self.yZ = yZ
        self.model_mah = model_mah
        self.model_reion = model_reion
        self.z_reion = z_reion
        self.M_c1_early = M_c1_early
        self.M_c1_0 = M_c1_0
        self.alpha1 = alpha1
        self.model_cooling = model_cooling
        self.M_c2_0 = M_c2_0
        self.alpha2 = alpha2
        self.model_sfr = model_sfr
        self.t_depl = t_depl
        self.R_loss = R_loss
        self.model_wind = model_wind
        self.eta_fixed = eta_fixed
        self.eta_min = eta_min
        self.eta_max = eta_max

    def dD_dt(self, t):
        z = self.cosmo.age(t, inverse=True)
        dD_dz = self.cosmo.growthFactor(z, derivative=1)
        dz_dt = self.cosmo.age(t, derivative=1, inverse=True)
        return dD_dz * dz_dt

    def dMh_dt(self, t, Mx_cur):
        z = self.cosmo.age(t, inverse=True)
        if self.model_mah == 'neistein08':
            dDdt = self.dD_dt(t)
            D = self.cosmo.growthFactor(z)
            return 1.06E12 * (Mx_cur[IH] / 1E12)**1.14 * dDdt / D**2
        else:
            raise ValueError('Unknown MAH model: %s' % self.model_mah)

    def eps_g1(self, t, Mx_cur):
        """Reionization efficiency factor."""
        if self.model_reion == 'none':
            return 1.0
        elif self.model_reion == 'okamoto08':
            z = self.cosmo.age(t, inverse=True)
            if z > self.z_reion:
                M_c1 = self.M_c1_early
            else:
                M_c1 = self.M_c1_0 * np.exp(-0.63 * z)
            alpha = self.alpha1
            return (1.0 + (2.0**(alpha / 3) - 1) * (Mx_cur[IH] / M_c1)**(-alpha))**(-3.0 / alpha)
        else:
            raise ValueError('Unknown reionization model: %s' % self.model_reion)

    def eps_g2(self, t, Mx_cur):
        """Cooling efficiency factor."""
        if self.model_cooling == 'none':
            return 1.0
        elif self.model_cooling == 'cutoff':
            z = self.cosmo.age(t, inverse=True)
            M_c2 = self.M_c2_0 * np.sqrt(self.cosmo.Ez(z)) * mass_so.deltaVir(z) / mass_so.deltaVir(0.0)
            alpha = self.alpha2
            return 1.0 - (1.0 + (2.0**(alpha / 3) - 1) * (Mx_cur[IH] / M_c2)**(-alpha))**(-3.0 / alpha)
        else:
            raise ValueError('Unknown cooling model: %s' % self.model_cooling)

    def sfr(self, t, Mx_cur):
        """Star formation rate."""
        if self.model_sfr == 'depl_time':
            return Mx_cur[IG] / self.t_depl
        else:
            raise ValueError('Unknown SFR model: %s' % self.model_sfr)

    def eta(self, t, Mx_cur):
        """Mass loading factor (wind strength)."""
        if self.model_wind == 'none':
            return 0.0
        elif self.model_wind == 'fixed':
            return self.eta_fixed
        elif self.model_wind == 'muratov15_star':
            ret = np.maximum(0.0, 3.6 * (np.maximum(Mx_cur[IS], 1E-20) / 1E10)**(-0.35) - 4.5)
            return np.clip(ret, self.eta_min, self.eta_max)
        else:
            raise ValueError('Unknown wind model: %s' % self.model_wind)

    def massRates(self, t, Mx_cur):
        """Coupled ODEs for [dMh/dt, dMg/dt, dMs/dt, dMZ/dt]."""
        dMh = self.dMh_dt(t, Mx_cur)
        eps1 = self.eps_g1(t, Mx_cur)
        eps2 = self.eps_g2(t, Mx_cur)
        eta_val = self.eta(t, Mx_cur)
        sfr_val = self.sfr(t, Mx_cur)

        dMg_in = dMh * self.fb * eps1 * eps2
        dMg_out = sfr_val * (1.0 - self.R_loss + eta_val)
        dMg = dMg_in - dMg_out
        dMs = sfr_val * (1.0 - self.R_loss)
        dMZ = dMg_in * self.Z_in + sfr_val * (
            (1.0 - self.R_loss) * self.yZ
            - (1.0 - self.R_loss + eta_val) * Mx_cur[IZ] / Mx_cur[IG])

        return [dMh, dMg, dMs, dMZ]

    def evolve(self, t_eval=None, z_eval=None, z_final=0.0):
        """Integrate the model from z_ini to z_final."""
        t_ini = self.cosmo.age(self.z_ini)
        t_final = self.cosmo.age(z_final)

        if t_eval is not None and z_eval is not None:
            raise ValueError('Specify t_eval or z_eval, not both.')
        elif z_eval is not None:
            t_eval = self.cosmo.age(z_eval)
        elif t_eval is None:
            t_eval = np.linspace(t_ini, t_final, 500)

        ret = scipy.integrate.solve_ivp(
            self.massRates, [t_ini, t_final], self.Mx_ini,
            t_eval=t_eval, rtol=1E-5, vectorized=False)

        t = ret['t']
        Mx = ret['y']
        z = self.cosmo.age(t, inverse=True)
        a = 1.0 / (1.0 + z)

        dMx = np.zeros_like(Mx)
        for i in range(len(t)):
            dMx[:, i] = self.massRates(t[i], Mx[:, i])

        return t, z, a, Mx, dMx

###################################################################################################
# Helper functions
###################################################################################################

def evaluateModel(z_ini, Mh_ini_array, **kwargs):
    """
    Run GalaxyModel for an array of initial halo masses.

    Returns:
        t, z, a : time/redshift/scale factor arrays (from first model)
        M_hist : array of shape (n_models, 4, n_times) with mass histories
        dM_hist : array of shape (n_models, 4, n_times) with rate histories
    """
    M_hist = []
    dM_hist = []
    t_, z_, a_ = None, None, None

    for j, Mh_ini in enumerate(Mh_ini_array):
        gm = GalaxyModel(z_ini, Mh_ini, **kwargs)
        t, z, a, Mx, dMx = gm.evolve()
        if j == 0:
            t_, z_, a_ = t, z, a
        M_hist.append(Mx)
        dM_hist.append(dMx)

    M_hist = np.array(M_hist)
    dM_hist = np.array(dM_hist)

    return t_, z_, a_, M_hist, dM_hist

###################################################################################################
# SHMR comparison functions
###################################################################################################

def mstarBehroozi13(Mhalo, z=0.0):
    """Behroozi+2013 SHMR fitting function. Mhalo in Msun, returns Mstar in Msun."""
    def bfz(x, z):
        a = 1.0 / (1.0 + z)
        nu = np.exp(-4.0 * a**2)
        alpha = -1.412 + nu * (0.731 * (a - 1.0))
        delta = 3.508 + nu * (2.608 * (a - 1.0) + (-0.043) * z)
        gamma = 0.316 + nu * (1.319 * (a - 1.0) + 0.279 * z)
        return (-np.log10(10**(alpha * x) + 1.0)
                + delta * (np.log10(1.0 + np.exp(x)))**gamma / (1.0 + np.exp(10**(-x))))

    a = 1.0 / (1.0 + z)
    nu = np.exp(-4.0 * a**2)
    log_M1 = 11.514 + nu * (-1.793 * (a - 1.0) + (-0.251) * z)
    log_eps = -1.777 + nu * (-0.006 * (a - 1.0) + 0.0 * z) + (-0.119) * (a - 1.0)
    log_Mstar = log_eps + log_M1 + bfz(np.log10(Mhalo) - log_M1, z) - bfz(0.0, z)
    return 10**log_Mstar

def shmrBehroozi13():
    """Return (Mvir, Mstar/Mvir) from Behroozi+2013 at z=0."""
    Mvir = 10**np.linspace(9.0, 16.0, 200)
    mstar = mstarBehroozi13(Mvir, 0.0)
    mask = (mstar >= 10**7.25)
    return Mvir[mask], mstar[mask] / Mvir[mask]

def mstarKravtsov18(Mvir):
    """Kravtsov+2018 SHMR at z=0. Mvir in Msun."""
    log_M1, log_eps, alpha, delta, gamma = 11.43, -1.663, -1.750, 4.290, 0.595
    def f(x):
        return (-np.log10(10**(alpha * x) + 1.0)
                + delta * (np.log10(1.0 + np.exp(x)))**gamma / (1.0 + np.exp(10**(-x))))
    M1 = 10**log_M1
    log_Mstar = np.log10(10**log_eps * M1) + f(np.log10(Mvir / M1)) - f(0.0)
    return 10**log_Mstar

def shmrKravtsov18():
    """Return (Mvir, Mstar/Mvir) from Kravtsov+2018 at z=0."""
    Mvir = 10**np.linspace(10.0, 15.2, 200)
    mstar = mstarKravtsov18(Mvir)
    return Mvir, mstar / Mvir

###################################################################################################
# Cosmology setup
###################################################################################################

cosmo = cosmology.setCosmology('planck18')

###################################################################################################
