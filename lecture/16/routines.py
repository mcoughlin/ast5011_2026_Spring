###################################################################################################
#
# AST 5011 - Lecture 17: The Formation of Dark Matter Halos
#
# Helper routines adapted from Benedikt Diemer's ASTR 620 course
#
###################################################################################################

import numpy as np
import h5py
from colossus.cosmology import cosmology

###################################################################################################

# Path to N-body simulation data (Erebos suite)
nbody_data_dir = '/Users/mcoughlin/Code/Teaching/AST5011/ast5011_2026_Spring_workspace/galaxies_course/data/'

###################################################################################################

def getMassFunction(sim_name, z, data_dir=None, min_n_ptl=200, n_bins=20, n_min_in_bin=20):
    """
    Compute the halo mass function from an Erebos simulation snapshot.

    Parameters
    ----------
    sim_name : str
        Simulation name (e.g. 'l0500-bol').
    z : float
        Redshift of the desired snapshot.
    data_dir : str
        Path to the data directory containing nbody/ subdirectory.
    min_n_ptl : int
        Minimum number of particles per halo.
    n_bins : int
        Number of logarithmic mass bins.
    n_min_in_bin : int
        Minimum number of halos per bin.

    Returns
    -------
    mf : ndarray
        dn/dlnM in h^3 Mpc^-3.
    bin_centers : ndarray
        Bin centers in h^-1 Msun.
    """
    if data_dir is None:
        data_dir = nbody_data_dir

    fn = data_dir + 'nbody/tree_%s.hdf5' % sim_name
    f = h5py.File(fn, 'r')
    snap_z = f['simulation'].attrs['snap_z']
    L_box = f['simulation'].attrs['box_size']
    m_ptl = f['simulation'].attrs['particle_mass']
    M_min = m_ptl * min_n_ptl
    V = L_box**3

    snap_idx = np.argmin(np.abs(snap_z - z))

    M = np.array(f['Mvir'][snap_idx, :])
    is_sub = np.array(f['is_subhalo'][snap_idx, :])
    f.close()

    mask = np.logical_not(is_sub) & (M >= M_min)
    M = M[mask]

    log_M = np.log(M)
    mf, bin_edges = np.histogram(log_M, bins=n_bins)
    bin_width = bin_edges[1:] - bin_edges[:-1]

    mask = (mf > n_min_in_bin)

    mf = np.array(mf, float) / bin_width / V
    bin_edges_log = np.log10(np.exp(bin_edges))
    bin_centers_log = 0.5 * (bin_edges_log[1:] + bin_edges_log[:-1])
    bin_centers = 10**bin_centers_log

    return mf[mask], bin_centers[mask]

###################################################################################################
# Milky Way parameters (Licquia & Newman 2016; Ninkovic 2017)
###################################################################################################

Rd_MW = 2.64  # kpc, disk scale length

###################################################################################################
# Cosmology setup
###################################################################################################

cosmo = cosmology.setCosmology('planck18')

###################################################################################################
