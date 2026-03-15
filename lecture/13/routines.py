###################################################################################################
#
# ASTR 620 - Galaxies
#
# obs_sdss.py: Routines related to SDSS data
#
# (c) Benedikt Diemer, University of Maryland (based on similar code by Andrey Kratsov)
#
###################################################################################################

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import urllib
from PIL import Image

from astropy.io import fits
from astropy.table import Table
from colossus.cosmology import cosmology
# Set the default cosmology
cosmo = cosmology.setCosmology('planck18')

# Save the default color cycle in a variable so we can set colors explicitly
prop_cycle = plt.rcParams['axes.prop_cycle']
color_cycle = prop_cycle.by_key()['color']

###################################################################################################

data_dir = './data'
sdss_img_dir = os.path.join(data_dir, 'sdss_images')
sdss_spec_dir = os.path.join(data_dir, 'sdss_spectra')

# The solid angle in radians covered by the spectroscopic survey
solid_angle = 9274.0 / (180.0 / np.pi)**2

# The limiting magnitude for our sample (Petrosian magnitude in r-band)
m_r_limit = 17.77

# The natural scale of arcsecond / pixel for the SDSS camera; 1/scale = 2.52 pixels / arcsec
sdss_pixel_scale = 0.396127

# Colors in which to plot ugriz filters
filter_colors = 'bgrmk'

# Approximate diameter of spectral fibers in arcseconds
sdss_fiber_size = 3.0

# Spectra were not taken for all galaxies above the limiting magnitude because of so-called
# fiber collisions, meaning that galaxies were too close on the sky to fit the spectroscopic
# fibers next to each other. This happened for about 7% of galaxies (Bernardi et al. 2010).
spectroscopic_completeness = 0.93

###################################################################################################

# Load the SDSS spectroscopic sample. There are a few galaxies with extremely low redshifts that
# can cause numerical issues when computing luminosity distances etc., so we remove those.

def loadSdssSpecSample():

    H5_1 = f"{data_dir}/sdss/sdss_part1.h5"
    H5_2 = f"{data_dir}/sdss/sdss_part2.h5"

    df1 = pd.read_hdf(H5_1, key='sdss')
    df2 = pd.read_hdf(H5_2, key='sdss')
    df = pd.concat([df1, df2], ignore_index=True)

    data = Table.from_pandas(df)

    mask = (data['z'] >= 1E-7)
    data = data[mask]

    return data

###################################################################################################

# Load the SDSS sample and extend the dataset by absolute magnitudes and colors including
# K-corrections, distance modulus, effective radii, and so on.

def loadSdssSpecSampleExtra():

    data_all = loadSdssSpecSample()

    # Mask: no crazy colors (they will throw the K-correction fits off) and no high redshifts
    # (where the fitting function is not calibrated)
    mg_raw = data_all['modelMag_g'] - data_all['extinction_g']
    mr_raw = data_all['modelMag_r'] - data_all['extinction_r']
    gr_raw = mg_raw - mr_raw
    mask = (gr_raw > -0.5) & (gr_raw < 2.0) & (data_all['z'] <= 0.6) \
             & (data_all['expRad_r'] > 0.0) & (data_all['deVRad_r'] > 0.0)
    data = data_all[mask]

    # Create dictionary. This will be temporary but it's easier to create the fields this way
    # and then add them to the main data array in the end.
    data2 = {}

    # Compute observed magnitudes, using model mags for color since they are measured using the
    # same apertures. Then use those colors as input to K-corrections.
    mu_raw = data['modelMag_u'] - data['extinction_u']
    mg_raw = data['modelMag_g'] - data['extinction_g']
    mr_raw = data['modelMag_r'] - data['extinction_r']
    mi_raw = data['modelMag_i'] - data['extinction_i']
    mz_raw = data['modelMag_z'] - data['extinction_z']
    ur_raw = mu_raw - mr_raw
    gr_raw = mg_raw - mr_raw
    gi_raw = mg_raw - mi_raw
    rz_raw = mr_raw - mz_raw
    data2['K_u'] = kCorrection('u', data['z'], 'u-r', ur_raw)
    data2['K_g'] = kCorrection('g', data['z'], 'g-r', gr_raw)
    data2['K_r'] = kCorrection('r', data['z'], 'g-r', gr_raw)
    data2['K_i'] = kCorrection('i', data['z'], 'g-i', gi_raw)
    data2['K_z'] = kCorrection('z', data['z'], 'r-z', rz_raw)

    # Compute K-corrected colors
    mu_cor = mu_raw - data2['K_u']
    mg_cor = mg_raw - data2['K_g']
    mr_cor = mr_raw - data2['K_r']
    mi_cor = mi_raw - data2['K_i']
    mz_cor = mz_raw - data2['K_z']
    data2['color_ug'] = mu_cor - mg_cor
    data2['color_gr'] = mg_cor - mr_cor
    data2['color_ri'] = mr_cor - mi_cor
    data2['color_iz'] = mi_cor - mz_cor

    # Compute distance and distance modulus; distances are in Mpc (not Mpc/h)
    cosmo = cosmology.getCurrent()
    data2['DM'] = cosmo.distanceModulus(data['z'])
    data2['dL'] = cosmo.luminosityDistance(data['z']) / cosmo.h
    data2['dA'] = cosmo.angularDiameterDistance(data['z']) / cosmo.h

    # Compute absolute magnitudes; for the r-band, we store the individual exp/deV mags as well
    for f in 'ugriz':
        offset = data['extinction_%c' % f] + data2['K_%c' % f] + data2['DM']
        tpes = ['model', 'cmodel', 'petro', 'fiber']
        if f == 'r':
            tpes.extend(['exp', 'deV'])
        for tpe in tpes:
            data2['M_%s_%c' % (tpe, f)] = data['%sMag_%c' % (tpe, f)] - offset
            data2['m_%s_%c' % (tpe, f)] = data['%sMag_%c' % (tpe, f)] - data['extinction_%c' % f]

    # Find the radius corresponding to the better fit between exponential and de Vaucouleurs
    diff_exp = np.abs(data['modelMag_r'] - data['expMag_r'])
    diff_dev = np.abs(data['modelMag_r'] - data['deVMag_r'])
    data2['mask_exp'] = (diff_exp < diff_dev)
    data2['Re_best'] = np.array(data['deVRad_r'])
    data2['Re_best'][data2['mask_exp']] = data['expRad_r'][data2['mask_exp']]
    kpc_factor = data2['dA'] * 1000.0 * np.pi / 180.0 / 3600.0
    data2['Re_best_kpc'] = data2['Re_best'] * kpc_factor

    # For the b/a factor, take the interpolated value between the exp and deV profiles
    data2['ab_best'] = data['fracdeV_r'] * data['deVAB_r'] + (1.0 - data['fracdeV_r']) * data['expAB_r']

    # Surface brightness. The factors of 2 in the logs comes from the fact that the half-light
    # radius, by construction, contains half the total flux from the galaxy.
    data2['mu_petro_r'] = data2['m_petro_r'] + 2.5 * np.log10(2.0 * np.pi * data['petroR50_r']**2)
    data2['mu_cmodel_r'] = data2['m_cmodel_r'] + 2.5 * np.log10(2.0 * np.pi * data2['Re_best']**2)

    # Concentration
    data2['c_90_50_r'] = data['petroR90_r'] / data['petroR50_r']
    # Add inverse Vmax. Here we do not impose a limit, which means that some nearby galaxies will
    # have extremely small Vmax and thus extremely large 1/Vmax weightings. It is best to excluse
    # such galaxies.
    data2['1/Vmax'] = inverseVmax(data2['M_petro_r'],m_limit = m_r_limit, z_max = None)

    for name, values in data2.items():
        data[name] = values  # Automatically adds or replaces the column
    return data

###################################################################################################

# Load the UPenn catalog of Meert et al. 2015. This function was adapted from Andrey Kravtsov's
# code. The photometric type determines which profile fits are loaded:
#
# 1 = best fit, 2 = deVaucouleurs, 3 = Sersic, 4 = DeVExp, 5 = SerExp

def loadUPennCatalog(phot_type = 3):

    def isSet(flag, bit):
        return (flag & (1 << bit)) != 0

    filenames = []
    filenames.append('UPenn_PhotDec_nonParam_rband.fits')
    filenames.append('UPenn_PhotDec_nonParam_gband.fits')
    filenames.append('UPenn_PhotDec_Models_rband.fits')
    filenames.append('UPenn_PhotDec_Models_gband.fits')
    filenames.append('UPenn_PhotDec_CAST.fits')
    filenames.append('UPenn_PhotDec_CASTmodels.fits')
    filenames.append('UPenn_PhotDec_H2011.fits')

    data_all = []
    names_in = []
    names_out = []
    dtypes = []
    for fn in filenames:
        file_path = os.path.join(data_dir, 'sdss_upenn', fn)
        if fn.startswith('UPenn_PhotDec_Models_'):
            d = fits.open(file_path)[phot_type].data
        else:
            d = fits.open(file_path)[1].data
        data_all.append(d)
        names_in.append(d.dtype.names)
        names_out_file = []
        if 'rband' in fn:
            field_ext = '_r'
        elif 'gband' in fn:
            field_ext = '_g'
        elif 'H2011' in fn:
            field_ext = '_h11'
        else:
            field_ext = ''
        for f in d.dtype.names:
            if f == 'objid':
                f_out = 'objID'
            else:
                f_out = f + field_ext
            names_out_file.append(f_out)
            dtypes.append((f_out, d[f].dtype))
        names_out.append(names_out_file)

    # Append extra fields
    dtypes.append(('dL', float))
    dtypes.append(('dA', float))
    dtypes.append(('DM', float))
    dtypes.append(('M_r', float))
    dtypes.append(('M_petro_r', float))
    dtypes.append(('1/Vmax', float))
    dtypes.append(('color_gr', float))

    # Create structured array and transfer data
    data = np.zeros((len(data_all[0])), dtype = dtypes)
    for i in range(len(data_all)):
        d = data_all[i]
        for j in range(len(names_in[i])):
            data[names_out[i][j]] = d[names_in[i][j]]

    # Minimal quality cuts and flags recommended by Alan Meert; taken from Andrey Kravtsov's code
    fflag = data['finalflag_r']
    mask_valid = (data['petroMag'] > 0.0) & (data['petroMag'] < 100.0)
    mask_valid &= (data['kcorr_r'] > 0)
    mask_valid &= (data['m_tot_r'] > 0) & (data['m_tot_r'] < 100)
    mask_valid &= (isSet(fflag, 1) | isSet(fflag, 4) | isSet(fflag, 10) | isSet(fflag, 14))
    data = data[mask_valid]

    # Compute extra fields
    data['dL'] = cosmo.luminosityDistance(data['z']) / cosmo.h
    data['dA'] = cosmo.angularDiameterDistance(data['z']) / cosmo.h
    data['DM'] = cosmo.distanceModulus(data['z'])
    data['M_r'] = data['m_tot_r'] - data['extinction_r'] - data['DM'] - data['kcorr_r']
    data['M_petro_r'] = data['petroMag'] - data['extinction_r'] - data['DM'] - data['kcorr_r']
    data['1/Vmax'] = inverseVmax(data['M_r'], m_limit = m_r_limit, z_max = None)
    data['color_gr'] = (data['m_tot_g'] - data['extinction_g']) - (data['m_tot_r'] - data['extinction_r'])

    return data

###################################################################################################

# This function loads SDSS surface density profiles from a FITS file. The table is documented as
# follows:
#
# bin      tinyint  1                        bin number (0..14)
# band     tinyint  1                        u,g,r,i,z (0..4)
# profMean real     4  nanomaggies/arcsec^2      Mean flux in annulus
# profErr  real     4  nanomaggies/arcsec^2      Standard deviation of mean pixel flux in annulus
# objID    bigint   8                        links to the photometric object

def loadSDSSProfiles():

    hdu = fits.open(os.path.join(data_dir,'sdss/sdss_profiles_dr8_m16.fit'))
    data = hdu[1].data

    # Remove all galaxies where the count is either below 10 bins (averaged over filters)
    # or above 15 bins (the maximum)
    ids, inv, counts_allfilt = np.unique(data['objID'], return_inverse = True, return_counts = True)
    counts_bins = 0.2 * counts_allfilt
    idxs_weird = np.where((counts_bins < 10.0) | (counts_allfilt > 5 * 15))[0]
    print('Removing %d/%d galaxies from the profiles set.' % (len(idxs_weird), len(ids)))
    mask_keep = np.ones_like(data['objID'], bool)
    for idx in idxs_weird:
        mask_keep[inv == idx] = False
    data = data[mask_keep]

    # Rearrange into 3D array of [galaxy, filter, radial bin]
    ids, inv, counts_allfilt = np.unique(data['objID'], return_inverse = True, return_counts = True)
    n_gal = len(ids)
    ar_prf = np.ones((n_gal, 5, 15), float) * -1
    ar_err = np.ones((n_gal, 5, 15), float) * -1
    ar_prf[inv, data['band'], data['bin']] = data['profMean']
    ar_err[inv, data['band'], data['bin']] = data['profErr']

    return ids, ar_prf, ar_err

###################################################################################################

# This function avoids the comovingDistance function in Colossus because it is slower than the
# other distances. We ignore any lower limit.

def inverseVmax(M_galaxies, m_limit = m_r_limit, z_max = None):

    dL = 10**(-5.0 + 0.2 * (m_limit - M_galaxies))
    z_lim = cosmo.luminosityDistance(dL * cosmo.h, inverse = True)
    if z_max is not None:
        z_lim = np.minimum(z_lim, z_max)
        dL = cosmo.luminosityDistance(z_lim)
    dC_max = dL / (1.0 + z_lim)
    Vmax = solid_angle / 3.0 * dC_max**3 / spectroscopic_completeness
    Vmax_inv_all = 1.0 / Vmax

    return Vmax_inv_all

###################################################################################################

# Compute the luminosity in solar units corresponding to an absolute magnitude definition.

def luminosity(data, mag_def = 'cmodel_r', evo_correction = True):

    M = data['M_%s' % mag_def]
    if evo_correction:
        M += 1.3 * data['z']

    filt = mag_def[-1]
    L = 10**(0.4 * (obs_utils.solar_mag[filt] - M))

    return L

###################################################################################################

# Function that checks whether a particular image cutout is already downloaded and does so if not.
# Technically, we would not need to add the object ID to the filename, but that way we can more
# easily identify which galaxy a cutout was meant to represent.
#
# The scale is the natural scale of SDSS in arcsec/pixel, according to
# https://skyserver.sdss.org/dr2/en/tools/chart/chart.asp

def getSdssImage(obj_id, ra, dec, scale = sdss_pixel_scale, n_pix = 200, grid = False):

    if n_pix < 64:
        print('WARNING: SDSS image server does not produce images with fewer than 64 pixels.')
        n_pix = 64

    if not os.path.exists(sdss_img_dir):
        os.makedirs(sdss_img_dir)

    fn = os.path.join(sdss_img_dir, '%d_ra_%.4f_dec_%.4f_scale_%.4f_w_%d_h_%d.jpg' \
            % (obj_id, ra, dec, scale, n_pix, n_pix))

    if not os.path.exists(fn):
        opt_str = ''
        if grid:
            opt_str += 'G'
        url = 'http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?ra=%.8f&dec=%.8f&scale=%.2f&width=%i&height=%i' \
                % (ra, dec, scale, n_pix, n_pix)
        if opt_str != '':
            url += '&opt=%s' % opt_str
        urllib.request.urlretrieve((url), fn)

    im = Image.open(fn)

    return im

###################################################################################################

# Function that checks whether a particular spectrum is already downloaded and does so if not.
# Technically, we would not need to add the object ID to the filename, but that way we can more
# easily identify which galaxy a spectrum was meant to represent.
#
# The normalization constant was taken from https://classic.sdss.org/dr7/products/spectra/, since
# the units should be erg/s/cm2/A.
#
# This routine was inspired by its equivalent in AstroML.

def getSdssSpectrum(obj_id, plate, mjd, fiber):

    if not os.path.exists(sdss_spec_dir):
        os.mkdir(sdss_spec_dir)

    fn = os.path.join(sdss_spec_dir, '%d_plate_%d_mjd_%d_fiber_%d.fit' % (obj_id, plate, mjd, fiber))

    if not os.path.exists(fn):
        url = 'http://das.sdss.org/spectro/1d_26/%04i/1d/spSpec-%05i-%04i-%03i.fit' \
                % (plate, mjd, plate, fiber)
        urllib.request.urlretrieve((url), fn)

    hdu = fits.open(fn)
    spectrum = hdu[0].data[0] * 1E-17
    coeff0 = hdu[0].header['COEFF0']
    coeff1 = hdu[0].header['COEFF1']
    lam = 10**(coeff0 + coeff1 * np.arange(len(spectrum)))

    return lam, spectrum

###################################################################################################

def imageCollage(sdss_objs, n_rows, n_cols, n_pix = 150, scale = sdss_pixel_scale, panel_size = 1.5,
                                save = False, show = True):

    fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * panel_size, n_rows * panel_size))

    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError('Please make sure your matplotlib can show jpg images.')

    for obj_id_, ra_, dec_, ax in zip(sdss_objs['objID'], sdss_objs['ra'], sdss_objs['dec'], axs.flatten()):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = getSdssImage(obj_id_, ra_, dec_, scale = scale, n_pix = n_pix)
        ax.imshow(im, origin = 'upper')
        ax.set_aspect('auto')

    fig.subplots_adjust(hspace = 0.02, wspace = 0.02)
    if save:
        plt.savefig('image_collage.jpg', bbox_inches = 'tight')
    elif show:
        plt.show()

    return fig, axs

###################################################################################################

# Create an image collage that shows example galaxies along two axes. If an inverted axis is
# desired (e.g., for magnitudes), pass the limits in the uninverted order (e.g., -22, -18) and
# set invert_x / invert_y to True.
#
# The function also overplots contours of the sample, which are typically derived from a larger
# sample than the images (which should probably be of nearby galaxies). Thus, the user can pass
# two separate masks for images and contours. If they are None, the entire sample is used.

def imageCollageVariables(data_all, var_x, var_y, label_x, label_y, min_x, max_x, min_y, max_y,
                                                invert_x = False, invert_y = False, n_bins_x = 10, n_bins_y = 10,
                                                mask_images = None, image_size_kpc = 20.0, random_seed = 2023,
                                                plot_contours = True, mask_contours = None,
                                                contour_levels = [0.99, 0.8, 0.6, 0.4, 0.2], contour_smoothing = 2.0,
                                                figsize = 8.0, save = False, show = True, fn_save = None):

    # ---------------------------------------------------------------------------------------------

    def pdf(x, hist, target):

        return np.sum(hist[hist > x]) - target

    # ---------------------------------------------------------------------------------------------

    def contourLabelFormat(x):

        idx = levels.index(x)
        lvl_pct = contour_levels[idx] * 100.0

        return r'%.0f\%%' % lvl_pct

    # ---------------------------------------------------------------------------------------------

    # Create bins
    bin_edges_x = np.linspace(min_x, max_x, n_bins_x + 1)
    bin_edges_y = np.linspace(min_y, max_y, n_bins_y + 1)
    dx = np.abs(max_x - min_x) / n_bins_x
    dy = np.abs(max_y - min_y) / n_bins_y

    # Apply mask if necessary
    if mask_images is None:
        data_im = data_all
    else:
        data_im = data_all[mask_images]

    # Create figure
    fig = plt.figure(figsize = (figsize, figsize))
    plt.subplots_adjust(left = 0.15, bottom = 0.15, right = 0.95, top = 0.95)
    ax = plt.gca()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    ax.tick_params(color = 'w', labelcolor = 'black', direction = 'in')
    ax.set_facecolor('k')
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if 'jpg' not in fig.canvas.get_supported_filetypes():
        raise ValueError('Please make sure your matplotlib can show jpg images.')

    # Go through bins
    np.random.seed(random_seed)
    for i in range(n_bins_x):
        mask_x = (data_im[var_x] >= bin_edges_x[i]) & (data_im[var_x] < bin_edges_x[i + 1])
        for j in range(n_bins_y):

            # Create sample for this bin and choose random galaxy
            mask_y = (data_im[var_y] >= bin_edges_y[j]) & (data_im[var_y] < bin_edges_y[j + 1])
            mask = mask_x & mask_y
            if np.count_nonzero(mask) == 0:
                continue
            idxs = np.where(mask)[0]
            idx = idxs[np.random.randint(0, len(idxs), 1)][0]

            # Get image of a particular size in kpc
            kpc_factor = data_im['dA'][idx] * 1000.0 * np.pi / 180.0 / 3600.0
            image_size_arcsec = image_size_kpc / kpc_factor
            image_size_pix = image_size_arcsec / sdss_pixel_scale
            image_size_pix = max(image_size_pix, 64)
            image_size_pix = min(image_size_pix, 256)
            image_size_pix = int(round(image_size_pix))
            im = getSdssImage(data_im['objID'][idx], data_im['ra'][idx], data_im['dec'][idx], n_pix = image_size_pix)

            # Plot
            extent = []
            if invert_x:
                extent.extend([bin_edges_x[i + 1], bin_edges_x[i]])
            else:
                extent.extend([bin_edges_x[i], bin_edges_x[i + 1]])
            if invert_y:
                extent.extend([bin_edges_y[j + 1], bin_edges_y[j]])
            else:
                extent.extend([bin_edges_y[j], bin_edges_y[j + 1]])
            plt.imshow(im, extent = extent, origin = 'upper')

    plt.gca().set_aspect(dx / dy)

    # Plot density contours if desired. We weight the distribution by 1/Vmax.
    if plot_contours:
        if mask_contours is None:
            data_ct = data_all
        else:
            data_ct = data_all[mask_contours]

        # Note that we still need to transpose the result of histogram2d, but not invert the
        # first axis as for imshow.
        hist, _, _ = np.histogram2d(data_ct[var_x], data_ct[var_y], bins = (50, 50),
                                                        range = [[min_x, max_x], [min_y, max_y]], weights = data_ct['1/Vmax'])
        hist /= np.sum(hist)
        hist = hist.T
        hist = scipy.ndimage.gaussian_filter(hist, contour_smoothing)

        # The levels are given as pdfs, as in, we want the contours that contain those fractions
        # of the sample. There does not seem to be an obvious way to do that with the contour()
        # function itself.
        levels = []
        for ct_level in contour_levels:
            lvl = scipy.optimize.brentq(pdf, 0.0, 1.0, args = (hist, ct_level))
            levels.append(lvl)

        # Plot contours and add labels. We need to trick the label function into not using the
        # actual contour values but the cumulative values passed to the function.
        cts = ax.contour(hist, extent = [min_x, max_x, min_y, max_y], levels = sorted(levels),
                        linewidths = 0.6, colors = 'w', alpha = 0.3)
        ax.clabel(cts, inline = True, fmt = contourLabelFormat, fontsize = 8)

    # Finalize plot
    if save:
        if fn_save is None:
            fn_save = 'image_collage_vars.pdf'
        plt.savefig(fn_save)
    elif show:
        plt.show()

    return fig

###################################################################################################

def imageAndSpectrum(sdss_obj, n_pix = 150, scale = sdss_pixel_scale, color_def = 'fiber',
                                        save = False, fn_out = None):

    if save:
        _, (ax0, ax1) = plt.subplots(1, 2, figsize = (9.0, 3.0))
        plt.subplots_adjust(wspace = 0.02)
    else:
        _, (ax0, ax1) = plt.subplots(1, 2, figsize = (12.0, 3.0))
        plt.subplots_adjust(wspace = -0.13)

    # Plot image
    im = getSdssImage(sdss_obj['objID'], sdss_obj['ra'], sdss_obj['dec'], scale = scale, n_pix = n_pix)
    ax0.axis('off')
    ax0.imshow(im, origin = 'upper')

    # Plot circle indicating the size of the spectral fiber
    r_fiber = sdss_fiber_size * 0.5 / scale
    circ = mpl.patches.Circle((n_pix * 0.5, n_pix * 0.5), r_fiber, color = 'w', fill = False, linestyle = 'solid', linewidth = 0.5)
    ax0.add_artist(circ)

    # Add text labels
    if color_def is not None:
        plt.text(0.95, 0.9, r'$(g-r)_{\rm %s} = %.2f$' % (color_def, sdss_obj['%sMag_g' % color_def] - sdss_obj['%sMag_r' % color_def]),
                transform = ax0.transAxes, fontsize = 14, ha = 'right', color = 'w')

    # Download SDSS spectrum using plate number, epoch, and fiber ID
    # Normalizing by a high percentile is better than max in the presence of spikes.
    lbda, F = getSdssSpectrum(sdss_obj['objID'], sdss_obj['plate'], sdss_obj['mjd'], sdss_obj['fiberID'])
    F_plot = 0.5 * F / np.percentile(F, 99.5)

    plt.sca(ax1)
    plt.xlim(3000, 10500)
    plt.ylim(0, 0.52)
    plt.xlabel(r'$\lambda\ ({\rm \AA})$')
    plt.ylabel(r'$S(\lambda)\ \mathrm{or}\ F(\lambda)\ \mathrm{[arbitrary\ units]}$')

    for f, c, loc in zip('ugriz', filter_colors, [3500, 4650, 6150, 7500, 8750]):
        fn = os.path.join(data_dir, 'sdss/filter_%c.txt' % (f))
        if not os.path.exists(fn):
            raise ValueError('Could not find file %s.' % fn)
        filt = np.loadtxt(fn, unpack = True)
        plt.fill(filt[0], filt[2], ec = c, fc = c, alpha = 0.4)
        plt.text(loc, 0.04, f, color = c, ha = 'center', va = 'center', fontsize = 14)

    # Plot spectrum
    ax1.plot(lbda, F_plot, '-', lw = 0.2, color = 'k')

    if save:
        if fn_out is None:
            fn_out = 'image_spec_%d.pdf' % (sdss_obj['objId'])
        plt.savefig(fn_out)
    else:
        plt.show()

    return

###################################################################################################

# Magnitude of the Sun in various filter bands (from Sparke & Gallagher Table 1.3)
solar_mag = {}
solar_mag['u'] = 6.55
solar_mag['g'] = 5.12
solar_mag['r'] = 4.68
solar_mag['i'] = 4.57
solar_mag['z'] = 4.60
solar_mag['bol'] = 4.74

# MW values from Tables 6/7 of Licquia & Newman 2016 (note the h factor in the magnitude)
Mstar_MW = 5.71E10
Mstar_MW_lo = Mstar_MW - 1.1E10
Mstar_MW_hi = Mstar_MW + 1.5E10

SFR_MW = 1.65
SFR_MW_lo = SFR_MW - 0.19
SFR_MW_hi = SFR_MW + 0.19

Mr_MW = -20.97 + 5.0 * np.log10(cosmo.h)
Mr_MW_lo = Mr_MW - 0.4
Mr_MW_hi = Mr_MW + 0.37

gr_MW = 0.678

Rd_MW = 2.64
Rd_MW_lo = Rd_MW - 0.13
Rd_MW_hi = Rd_MW + 0.13

Re_MW = 1.678 * Rd_MW
Re_MW_lo = 1.678 * Rd_MW_lo
Re_MW_hi = 1.678 * Rd_MW_hi

###################################################################################################

# Rotate the standard RA/DEC coordinate system into supergalactic coordinates. The function
# expects an array of shape [3, N] and returns the same dimensions.

def superGalacticCoordinates(xyz):

    M_rot = np.array([[0.3751891698, 0.3408758302, 0.8619957978],
                                    [-0.8982988298, -0.0957026824, 0.4288358766],
                                    [0.2286750954, -0.9352243929, 0.2703017493]])

    return np.dot(M_rot, xyz)

###################################################################################################

# K-corrections calculator in Python by Chilingarian, Melchior & Zolotukhin 2012. Available
# filter-colour combinations must be present in the coeff dictionary keys.
#
# filter_name:    Name of the filter to calculate K-correction for, e.g. 'u', 'g', 'r' for some of
#                 the SDSS filters, or 'J2', 'H2', 'Ks2' for 2MASS filters (must be present in
#                 `coeff` dictionary)
#
# redshift:       Redshift of a galaxy, should be between 0.0 and 0.5
#
# colour_name:    Human name of the colour, e.g. 'u - g', 'g - r', 'V - Rc', 'J2 - Ks2'
#                 (must be present in `coeff` dictionary)
#
# colour_value:   Value of the galaxy's colour, specified in colour_name

def kCorrection(filter_name, redshift, colour_name, colour_value):

    coeff = {

            'B_BRc': [
                [0, 0, 0, 0],
                [-1.99412, 3.45377, 0.818214, -0.630543],
                [15.9592, -3.99873, 6.44175, 0.828667],
                [-101.876, -44.4243, -12.6224, 0],
                [299.29, 86.789, 0, 0],
                [-304.526, 0, 0, 0],
            ],

            'B_BIc': [
                [0, 0, 0, 0],
                [2.11655, -5.28948, 4.5095, -0.8891],
                [24.0499, -4.76477, -1.55617, 1.85361],
                [-121.96, 7.73146, -17.1605, 0],
                [236.222, 76.5863, 0, 0],
                [-281.824, 0, 0, 0],
            ],

            'H2_H2Ks2': [
                [0, 0, 0, 0],
                [-1.88351, 1.19742, 10.0062, -18.0133],
                [11.1068, 20.6816, -16.6483, 139.907],
                [-79.1256, -406.065, -48.6619, -430.432],
                [551.385, 1453.82, 354.176, 473.859],
                [-1728.49, -1785.33, -705.044, 0],
                [2027.48, 950.465, 0, 0],
                [-741.198, 0, 0, 0],
            ],

            'H2_J2H2': [
                [0, 0, 0, 0],
                [-4.99539, 5.79815, 4.19097, -7.36237],
                [70.4664, -202.698, 244.798, -65.7179],
                [-142.831, 553.379, -1247.8, 574.124],
                [-414.164, 1206.23, 467.602, -799.626],
                [763.857, -2270.69, 1845.38, 0],
                [-563.812, -1227.82, 0, 0],
                [1392.67, 0, 0, 0],
            ],

            'Ic_VIc': [
                [0, 0, 0, 0],
                [-7.92467, 17.6389, -15.2414, 5.12562],
                [15.7555, -1.99263, 10.663, -10.8329],
                [-88.0145, -42.9575, 46.7401, 0],
                [266.377, -67.5785, 0, 0],
                [-164.217, 0, 0, 0],
            ],

            'J2_J2Ks2': [
                [0, 0, 0, 0],
                [-2.85079, 1.7402, 0.754404, -0.41967],
                [24.1679, -34.9114, 11.6095, 0.691538],
                [-32.3501, 59.9733, -29.6886, 0],
                [-30.2249, 43.3261, 0, 0],
                [-36.8587, 0, 0, 0],
            ],

            'J2_J2H2': [
                [0, 0, 0, 0],
                [-0.905709, -4.17058, 11.5452, -7.7345],
                [5.38206, -6.73039, -5.94359, 20.5753],
                [-5.99575, 32.9624, -72.08, 0],
                [-19.9099, 92.1681, 0, 0],
                [-45.7148, 0, 0, 0],
            ],

            'Ks2_J2Ks2': [
                [0, 0, 0, 0],
                [-5.08065, -0.15919, 4.15442, -0.794224],
                [62.8862, -61.9293, -2.11406, 1.56637],
                [-191.117, 212.626, -15.1137, 0],
                [116.797, -151.833, 0, 0],
                [41.4071, 0, 0, 0],
            ],

            'Ks2_H2Ks2': [
                [0, 0, 0, 0],
                [-3.90879, 5.05938, 10.5434, -10.9614],
                [23.6036, -97.0952, 14.0686, 28.994],
                [-44.4514, 266.242, -108.639, 0],
                [-15.8337, -117.61, 0, 0],
                [28.3737, 0, 0, 0],
            ],

            'Rc_BRc': [
                [0, 0, 0, 0],
                [-2.83216, 4.64989, -2.86494, 0.90422],
                [4.97464, 5.34587, 0.408024, -2.47204],
                [-57.3361, -30.3302, 18.4741, 0],
                [224.219, -19.3575, 0, 0],
                [-194.829, 0, 0, 0],
            ],

            'Rc_VRc': [
                [0, 0, 0, 0],
                [-3.39312, 16.7423, -29.0396, 25.7662],
                [5.88415, 6.02901, -5.07557, -66.1624],
                [-50.654, -13.1229, 188.091, 0],
                [131.682, -191.427, 0, 0],
                [-36.9821, 0, 0, 0],
            ],

            'U_URc': [
                [0, 0, 0, 0],
                [2.84791, 2.31564, -0.411492, -0.0362256],
                [-18.8238, 13.2852, 6.74212, -2.16222],
                [-307.885, -124.303, -9.92117, 12.7453],
                [3040.57, 428.811, -124.492, -14.3232],
                [-10677.7, -39.2842, 197.445, 0],
                [16022.4, -641.309, 0, 0],
                [-8586.18, 0, 0, 0],
            ],

            'V_VIc': [
                [0, 0, 0, 0],
                [-1.37734, -1.3982, 4.76093, -1.59598],
                [19.0533, -17.9194, 8.32856, 0.622176],
                [-86.9899, -13.6809, -9.25747, 0],
                [305.09, 39.4246, 0, 0],
                [-324.357, 0, 0, 0],
            ],

            'V_VRc': [
                [0, 0, 0, 0],
                [-2.21628, 8.32648, -7.8023, 9.53426],
                [13.136, -1.18745, 3.66083, -41.3694],
                [-117.152, -28.1502, 116.992, 0],
                [365.049, -93.68, 0, 0],
                [-298.582, 0, 0, 0],
            ],

            'FUV_FUVNUV': [
                [0, 0, 0, 0],
                [-0.866758, 0.2405, 0.155007, 0.0807314],
                [-1.17598, 6.90712, 3.72288, -4.25468],
                [135.006, -56.4344, -1.19312, 25.8617],
                [-1294.67, 245.759, -84.6163, -40.8712],
                [4992.29, -477.139, 174.281, 0],
                [-8606.6, 316.571, 0, 0],
                [5504.2, 0, 0, 0],
            ],

            'FUV_FUVu': [
                [0, 0, 0, 0],
                [-1.67589, 0.447786, 0.369919, -0.0954247],
                [2.10419, 6.49129, -2.54751, 0.177888],
                [15.6521, -32.2339, 4.4459, 0],
                [-48.3912, 37.1325, 0, 0],
                [37.0269, 0, 0, 0],
            ],

            'g_gi': [
                [0, 0, 0, 0],
                [1.59269, -2.97991, 7.31089, -3.46913],
                [-27.5631, -9.89034, 15.4693, 6.53131],
                [161.969, -76.171, -56.1923, 0],
                [-204.457, 217.977, 0, 0],
                [-50.6269, 0, 0, 0],
            ],

            'g_gz': [
                [0, 0, 0, 0],
                [2.37454, -4.39943, 7.29383, -2.90691],
                [-28.7217, -20.7783, 18.3055, 5.04468],
                [220.097, -81.883, -55.8349, 0],
                [-290.86, 253.677, 0, 0],
                [-73.5316, 0, 0, 0],
            ],

            'g_gr': [
                [0, 0, 0, 0],
                [-2.45204, 4.10188, 10.5258, -13.5889],
                [56.7969, -140.913, 144.572, 57.2155],
                [-466.949, 222.789, -917.46, -78.0591],
                [2906.77, 1500.8, 1689.97, 30.889],
                [-10453.7, -4419.56, -1011.01, 0],
                [17568, 3236.68, 0, 0],
                [-10820.7, 0, 0, 0],
            ],

            'H_JH': [
                [0, 0, 0, 0],
                [-1.6196, 3.55254, 1.01414, -1.88023],
                [38.4753, -8.9772, -139.021, 15.4588],
                [-417.861, 89.1454, 808.928, -18.9682],
                [2127.81, -405.755, -1710.95, -14.4226],
                [-5719, 731.135, 1284.35, 0],
                [7813.57, -500.95, 0, 0],
                [-4248.19, 0, 0, 0],
            ],

            'H_HK': [
                [0, 0, 0, 0],
                [0.812404, 7.74956, 1.43107, -10.3853],
                [-23.6812, -235.584, -147.582, 188.064],
                [283.702, 2065.89, 721.859, -713.536],
                [-1697.78, -7454.39, -1100.02, 753.04],
                [5076.66, 11997.5, 460.328, 0],
                [-7352.86, -7166.83, 0, 0],
                [4125.88, 0, 0, 0],
            ],

            'i_gi': [
                [0, 0, 0, 0],
                [-2.21853, 3.94007, 0.678402, -1.24751],
                [-15.7929, -19.3587, 15.0137, 2.27779],
                [118.791, -40.0709, -30.6727, 0],
                [-134.571, 125.799, 0, 0],
                [-55.4483, 0, 0, 0],
            ],

            'i_ui': [
                [0, 0, 0, 0],
                [-3.91949, 3.20431, -0.431124, -0.000912813],
                [-14.776, -6.56405, 1.15975, 0.0429679],
                [135.273, -1.30583, -1.81687, 0],
                [-264.69, 15.2846, 0, 0],
                [142.624, 0, 0, 0],
            ],

            'J_JH': [
                [0, 0, 0, 0],
                [0.129195, 1.57243, -2.79362, -0.177462],
                [-15.9071, -2.22557, -12.3799, -2.14159],
                [89.1236, 65.4377, 36.9197, 0],
                [-209.27, -123.252, 0, 0],
                [180.138, 0, 0, 0],
            ],

            'J_JK': [
                [0, 0, 0, 0],
                [0.0772766, 2.17962, -4.23473, -0.175053],
                [-13.9606, -19.998, 22.5939, -3.99985],
                [97.1195, 90.4465, -21.6729, 0],
                [-283.153, -106.138, 0, 0],
                [272.291, 0, 0, 0],
            ],

            'K_HK': [
                [0, 0, 0, 0],
                [-2.83918, -2.60467, -8.80285, -1.62272],
                [14.0271, 17.5133, 42.3171, 4.8453],
                [-77.5591, -28.7242, -54.0153, 0],
                [186.489, 10.6493, 0, 0],
                [-146.186, 0, 0, 0],
            ],

            'K_JK': [
                [0, 0, 0, 0],
                [-2.58706, 1.27843, -5.17966, 2.08137],
                [9.63191, -4.8383, 19.1588, -5.97411],
                [-55.0642, 13.0179, -14.3262, 0],
                [131.866, -13.6557, 0, 0],
                [-101.445, 0, 0, 0],
            ],

            'NUV_NUVr': [
                [0, 0, 0, 0],
                [2.2112, -1.2776, 0.219084, 0.0181984],
                [-25.0673, 5.02341, -0.759049, -0.0652431],
                [115.613, -5.18613, 1.78492, 0],
                [-278.442, -5.48893, 0, 0],
                [261.478, 0, 0, 0],
            ],

            'NUV_NUVg': [
                [0, 0, 0, 0],
                [2.60443, -2.04106, 0.52215, 0.00028771],
                [-24.6891, 5.70907, -0.552946, -0.131456],
                [95.908, -0.524918, 1.28406, 0],
                [-208.296, -10.2545, 0, 0],
                [186.442, 0, 0, 0],
            ],

            'r_gr': [
                [0, 0, 0, 0],
                [1.83285, -2.71446, 4.97336, -3.66864],
                [-19.7595, 10.5033, 18.8196, 6.07785],
                [33.6059, -120.713, -49.299, 0],
                [144.371, 216.453, 0, 0],
                [-295.39, 0, 0, 0],
            ],

            'r_ur': [
                [0, 0, 0, 0],
                [3.03458, -1.50775, 0.576228, -0.0754155],
                [-47.8362, 19.0053, -3.15116, 0.286009],
                [154.986, -35.6633, 1.09562, 0],
                [-188.094, 28.1876, 0, 0],
                [68.9867, 0, 0, 0],
            ],

            'u_ur': [
                [0, 0, 0, 0],
                [10.3686, -6.12658, 2.58748, -0.299322],
                [-138.069, 45.0511, -10.8074, 0.95854],
                [540.494, -43.7644, 3.84259, 0],
                [-1005.28, 10.9763, 0, 0],
                [710.482, 0, 0, 0],
            ],

            'u_ui': [
                [0, 0, 0, 0],
                [11.0679, -6.43368, 2.4874, -0.276358],
                [-134.36, 36.0764, -8.06881, 0.788515],
                [528.447, -26.7358, 0.324884, 0],
                [-1023.1, 13.8118, 0, 0],
                [721.096, 0, 0, 0],
            ],

            'u_uz': [
                [0, 0, 0, 0],
                [11.9853, -6.71644, 2.31366, -0.234388],
                [-137.024, 35.7475, -7.48653, 0.655665],
                [519.365, -20.9797, 0.670477, 0],
                [-1028.36, 2.79717, 0, 0],
                [767.552, 0, 0, 0],
            ],

            'Y_YH': [
                [0, 0, 0, 0],
                [-2.81404, 10.7397, -0.869515, -11.7591],
                [10.0424, -58.4924, 49.2106, 23.6013],
                [-0.311944, 84.2151, -100.625, 0],
                [-45.306, 3.77161, 0, 0],
                [41.1134, 0, 0, 0],
            ],

            'Y_YK': [
                [0, 0, 0, 0],
                [-0.516651, 6.86141, -9.80894, -0.410825],
                [-3.90566, -4.42593, 51.4649, -2.86695],
                [-5.38413, -68.218, -50.5315, 0],
                [57.4445, 97.2834, 0, 0],
                [-64.6172, 0, 0, 0],
            ],

            'z_gz': [
                [0, 0, 0, 0],
                [0.30146, -0.623614, 1.40008, -0.534053],
                [-10.9584, -4.515, 2.17456, 0.913877],
                [66.0541, 4.18323, -8.42098, 0],
                [-169.494, 14.5628, 0, 0],
                [144.021, 0, 0, 0],
            ],

            'z_rz': [
                [0, 0, 0, 0],
                [0.669031, -3.08016, 9.87081, -7.07135],
                [-18.6165, 8.24314, -14.2716, 13.8663],
                [94.1113, 11.2971, -11.9588, 0],
                [-225.428, -17.8509, 0, 0],
                [197.505, 0, 0, 0],
            ],

            'z_uz': [
                [0, 0, 0, 0],
                [0.623441, -0.293199, 0.16293, -0.0134639],
                [-21.567, 5.93194, -1.41235, 0.0714143],
                [82.8481, -0.245694, 0.849976, 0],
                [-185.812, -7.9729, 0, 0],
                [168.691, 0, 0, 0],
            ],

            }

    cname = colour_name.replace('-', '')
    cname = cname.replace(' ', '')
    c = coeff[filter_name + '_' + cname]
    kcor = 0.0

    for x, _ in enumerate(c):
        for y, _ in enumerate(c[x]):
            kcor += c[x][y] * redshift ** x * colour_value ** y

    return kcor

###################################################################################################
