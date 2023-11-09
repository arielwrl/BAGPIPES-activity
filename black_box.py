import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import bagpipes as pipes
import seaborn as sns

sns.set_style('ticks')


def make_bagpipes_model(age, tau, mass, Av, z, filter_list=None):
    """

    Plots a BAGPIPES model for HST filters

    """

    exp = {}
    exp["age"] = age
    exp["tau"] = tau
    exp["massformed"] = mass
    exp["metallicity"] = 2

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = Av
    dust["eta"] = 2.

    nebular = {}
    nebular["logU"] = -3.

    model_components = {}
    model_components["redshift"] = z
    model_components["delayed"] = exp
    model_components["dust"] = dust
    model_components["t_bc"] = 0.02
    # model_components["nebular"] = nebular

    if filter_list is not None:
        model = pipes.model_galaxy(model_components, filt_list=filter_list,
                                   spec_wavs=np.arange(2400, 9100, 1))
    else:
        model = pipes.model_galaxy(model_components, spec_wavs=np.arange(2400, 9100, 1))

    return model


def plot_filters(paths_to_filters, filter_names):
    colors = sns.color_palette('plasma', 5)

    plt.figure(figsize=(9, 5))

    for i in range(len(paths_to_filters)):
        filter = paths_to_filters[i]

        wl, transmission = np.genfromtxt(filter).transpose()
        plt.plot(wl, transmission, color=colors[i], lw=3.5, label=filter_names[i])
        plt.fill_between(wl, np.zeros_like(transmission), transmission,
                         color=colors[i], alpha=0.2)

    plt.legend(frameon=False, fontsize=16)

    plt.ylim(0)

    plt.xlabel(r'$\lambda \mathrm{[\AA]}$', fontsize=20)
    plt.ylabel(r'$T_\lambda$', fontsize=20)


def plot_model(age, tau, mass, Av=0, redshift=0, filters=None):
    if filters is None:
        model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av,
                                    z=redshift)
    else:
        model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av,
                                    z=redshift, filter_list=filters)

    wave_limit = (model.wavelengths > 2000) & (model.wavelengths < 9000)

    plt.figure(figsize=(13, 7))
    plt.plot(model.wavelengths * (1 + redshift), model.spectrum_full,
             color='mediumvioletred', label='Model Spectrum', lw=0.75)

    if filters is not None:
        plt.scatter(model.filter_set.eff_wavs, model.photometry, s=200,
                    c='indigo', edgecolors='white',
                    label='Model Photometry', zorder=10)

    plt.xlim(2000, 9000)
    plt.ylim(0, np.max(model.spectrum_full[wave_limit]))

    plt.xlabel(r'$\lambda \mathrm{[\AA]}$', fontsize=16)
    plt.ylabel(r'$F_\lambda \mathrm{[erg \, cm^{-2} \, s^{-1} \, \AA^{-1}]}$',
               fontsize=16)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                         useMathText=True)

    return model


def plot_model_unphysical(age, tau, mass, Av=0, redshift=0):
    model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av, z=0)

    wave_limit = (model.wavelengths > 2000) & (model.wavelengths < 9000)

    plt.figure(figsize=(13, 7))
    plt.plot(model.wavelengths * (1 + redshift), model.spectrum_full,
             color='mediumvioletred', label='Model Spectrum', lw=0.75)

    plt.xlim(2000, 9000)
    plt.ylim(0, np.max(model.spectrum_full[wave_limit]))

    plt.xlabel(r'$\lambda \mathrm{[\AA]}$', fontsize=16)
    plt.ylabel(r'$F_\lambda \mathrm{[erg \, cm^{-2} \, s^{-1} \, \AA^{-1}]}$',
               fontsize=16)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                         useMathText=True)

    return model


def plot_sfh(age, tau, mass, redshift=0, Av=0):
    model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av, z=redshift)

    plt.figure(figsize=(8, 7))

    plt.plot(13.5 - model.sfh.ages / 1e9, model.sfh.sfh, '-k')
    plt.xlim(0, 13.5)

    plt.xlabel('Age of the Universe', fontsize=16)
    plt.ylabel('Star-formation Rate', fontsize=16)

    if (np.max(model.sfh.sfh) < 1) | (np.max(model.sfh.sfh) > 100):
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                             useMathText=True)


def plot_sfh_advanced(age, tau, mass, redshift=0, Av=0, color='k', label=None, ax=None):
    
    if ax is None:
        ax = plt.gca()

    model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av, z=redshift)

    ax.plot(13.5 - model.sfh.ages / 1e9, model.sfh.sfh, color=color, label=label)
    ax.set_xlim(0, 13.5)

    ax.set_xlabel('Age of the Universe', fontsize=16)
    ax.set_ylabel('Star-formation Rate', fontsize=16)

    if (np.max(model.sfh.sfh) < 1) | (np.max(model.sfh.sfh) > 100):
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                             useMathText=True)


def read_hst_photometry(region, data_dir='/content/drive/MyDrive/PCTO_BAGPIPES/Data/'):
    """

    Reads HST photometry

    """

    flux, error = np.genfromtxt(data_dir + 'JO175_' + region +
                                '_PCTO.dat').transpose()

    return flux, error


def resampler(x_old, y_old, x_new):
    interp = interp1d(x_old, y_old, bounds_error=False
                      , fill_value=(0., 0.))

    y_new = interp(x_new)

    return y_new


def pivot_wavelength(filter_curve):
    # Reading filter if needed:
    if type(filter_curve) is str:
        wl_filter, T = np.genfromtxt(filter_curve).transpose()
    else:
        wl_filter, T = filter_curve[0], filter_curve[1]

    # Calculating pivot_wavelength
    pivot_wl = np.trapz(T * wl_filter, dx=1) / np.trapz(T * (wl_filter ** -1), dx=1)
    pivot_wl = np.sqrt(pivot_wl)

    return pivot_wl


def plot_observation(region, filters,
                     data_dir='/content/drive/MyDrive/PCTO_BAGPIPES/Data/',
                     show_filters=False):
    """

    Plots HST observation

    """

    flux, error = read_hst_photometry(region)

    wavelengths = np.array([pivot_wavelength(filter) for filter in filters])

    plt.figure(figsize=(13, 7))

    plt.errorbar(wavelengths, flux, yerr=error, fmt='o', ms=10,
                 color='firebrick', label='Observed Photometry')

    if show_filters:

        for filter in filters:
            wl_filter, T = np.genfromtxt(filter).transpose()
            plt.plot(wl_filter, 1.3 * T * np.max(flux), '--k')

    plt.legend(frameon=True, fontsize=14)

    plt.xlim(2000, 9000)
    plt.ylim(0, 1.3 * np.max(flux))

    plt.xlabel(r'$\lambda \mathrm{[\AA]}$', fontsize=16)
    plt.ylabel(r'$F_\lambda \mathrm{[erg \, cm^{-2} \, s^{-1} \, \AA^{-1}]}$', fontsize=16)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                         useMathText=True)


def plot_model_and_observation(region, age, tau, mass, Av, redshift, filters,
                               data_dir='/content/drive/MyDrive/PCTO_BAGPIPES/Data/'):
    """

    Overplots BAGPIPES model and HST observation

    """

    if filters is None:
        model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av,
                                    z=redshift)
    else:
        model = make_bagpipes_model(age=age, tau=tau, mass=mass, Av=Av,
                                    z=redshift, filter_list=filters)

    flux, error = read_hst_photometry(region, data_dir)

    wave_limit = (model.wavelengths > 2000) & (model.wavelengths < 9000)
    redshifted_wave_limit = (model.wavelengths * (1 + redshift) > 2000) & (model.wavelengths * (1 + redshift) < 9000)

    plt.figure(figsize=(13, 7))
    plt.plot(model.wavelengths * (1 + redshift), model.spectrum_full,
             color='mediumvioletred', label='Model Spectrum', lw=0.75)

    if filters is not None:
        plt.scatter(model.filter_set.eff_wavs, model.photometry, s=200,
                    c='indigo', edgecolors='white',
                    label='Model Photometry', zorder=10)

    plt.scatter(model.filter_set.eff_wavs, flux, facecolors='none',
                edgecolors='k', s=300, linewidths=3,
                label='Observed Photometry', zorder=5)

    plt.xlim(2000, 9000)
    plt.ylim(0, np.max(model.spectrum_full[redshifted_wave_limit]))

    plt.xlabel(r'$\lambda \mathrm{[\AA]}$', fontsize=16)
    plt.ylabel(r'$F_\lambda \mathrm{[erg \, cm^{-2} \, s^{-1} \, \AA^{-1}]}$',
               fontsize=16)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                         useMathText=True)

    plt.legend(frameon=False, fontsize=14)

    if np.max(model.spectrum_full[redshifted_wave_limit]) < np.max(flux):
        print('WARNING: Your observed fluxes dissappeared because they are out of the scale of the plot!')

    return model
