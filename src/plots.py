#!/usr/bin/env python3

"""
This python module contains plot utilities used to generate 
all plots shown in the Jannsen+2026.
"""

# Built-in
import shutil
import datetime

# Dependencies
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from jax import numpy as jnp

# Internal dependencies
import smbhb as bh
import utils as ut

#--------------------------------------------------------------#
#                        PUBLIC METHODS                        #
#--------------------------------------------------------------#

def setup(warning=True):
    """Default matplotlibrc settings for plotting.
    """    
    # Control tickers
    plt.rcParams['xtick.top']           = True
    plt.rcParams['xtick.bottom']        = True
    plt.rcParams['xtick.top']           = True
    plt.rcParams['xtick.labelbottom']   = True
    plt.rcParams['xtick.direction']     = 'out'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top']     = True
    plt.rcParams['xtick.minor.top']     = True
    plt.rcParams['xtick.minor.bottom']  = True
    plt.rcParams['xtick.alignment']     = 'center'
    plt.rcParams['ytick.left']          = True
    plt.rcParams['ytick.right']         = True
    plt.rcParams['ytick.labelleft']     = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.left']    = True
    plt.rcParams['ytick.major.right']   = True
    plt.rcParams['ytick.minor.left']    = True
    plt.rcParams['ytick.minor.right']   = True
    # Legends
    plt.rcParams['legend.loc']        = 'best'
    plt.rcParams['legend.frameon']    = True
    plt.rcParams['legend.fancybox']   = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize']   = 17
    # Font
    plt.rcParams['font.family']    = 'serif'
    plt.rcParams['font.size']      = 17
    plt.rcParams['axes.titlesize'] = 17
    # Use LaTeX if installed
    if shutil.which('latex'):
        plt.rcParams['text.usetex'] = True
    else:
        plt.rcParams['text.usetex'] = False 
    # Ignore warnings if requested
    if not warning:
        warnings.simplefilter("ignore")

        
def plot_aitoff(df_agn, df_all=False, df_lop=False, df_best=False, NED=False):
    """Function to generate plot galactic aFetch Gaia info for each source in data frame.    
    """    
    if df_best is not False:
        df = df_best
    elif df_lop is not False:
        df = df_lop
    elif df_all is not False:
        df = df_all
    else:
        df = df_agn
    title = (f'Total: {df.shape[0]}, ' +
             f'LOPN1: {df[df.b > 0].shape[0]}, ' + 
             f'LOPS2: {df[df.b < 0].shape[0]}')
    # Plot PLATO AGNs
    fig, ax = pt.drawStarsInSkyAitoff(
        df_agn.ra, df_agn.dec, column=df_agn.ncam, cbarMap='Blues',
        cbarLabel=r'N-CAM visibility, $n_{\rm NCAM}$',
        title=title, fs=13, figsize=(10,7))
    # Plot all candidates
    if df_all is not False:
        if NED:
            ra, dec = df_all.RA, df_all.Dec
        else:
            ra, dec = df_all.ra, df_all.dec
        gal = SkyCoord(ra, dec, frame='icrs', unit=u.deg).galactic
        ax.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian,
                   c='orange', marker='o', s=10, ec='w', lw=0.8, zorder=4)
    # Plot candidates within LOPs
    if df_lop is not False:
        gal = SkyCoord(ra, dec, frame='icrs', unit=u.deg).galactic
        ax.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian,
                   c='k', marker='o', s=20, ec='w', lw=0.8, zorder=5)
    # Plot best candidates within LOPs
    if df_best is not False:
        gal = SkyCoord(df_best.ra, df_best.dec, frame='icrs', unit=u.deg).galactic
        ax.scatter(-gal.l.wrap_at('180d').radian, gal.b.radian,
                   c='limegreen', marker='o', s=20, ec='w', lw=0.8, zorder=5);
    return fig, ax


def plot_quarter_marks(ax, time, N, Q0=None):
    """Plot combined or components of model.
    """
    quarter = ut.quarter()
    n_quarter = int(np.ceil(time[-1] / quarter))
    quarters = []
    position = []
    for q in range(1, n_quarter+1):
        time_Q = q * quarter
        xpos = time_Q - quarter / 2
        quarters.append(q)
        position.append(xpos)
        for i in range(N):
            ax[i].axvline(x=time_Q, c='k', linestyle='--', lw=0.5, alpha=0.5, zorder=0)
    # Plot marks on top y-axis
    if Q0 is not None:
        ax0 = ax[0].twiny()
        ax0.set_xticks(position)
        ax0.tick_params(axis='x', which='major', labelsize=15)
        ax0.set_xticklabels([f'Q{q + Q0-1}' for q in quarters])
        ax0.set_xlim(time.min(), time.max())
        ax0.xaxis.set_minor_locator(ticker.NullLocator())


def plot_model(df, lw=1.5, figsize=(9,5)):
    """Plot combined or components of model.
    """
    time = df.time.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if 'flux_red' in df:
        plt.plot(time, df.flux_red, color='tomato', label="DRW", lw=lw/2)
    if 'flux_boost' in df:
        ax.plot(time, df.flux_boost, c='orange', label="Boosting", lw=lw)
    if 'flux_lens' in df:
        ax.plot(time, df.flux_lens, c='royalblue', label="Lensing", lw=lw)
    if 'flux' in df:        
        ax.plot(time, df.flux, c='k', label="Model", lw=lw/2)
    ax.set_xlabel(r"Time [day]")
    ax.set_ylabel(r"Relative flux")
    ax.set_xlim(time[0], time[-1])
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_lc(df, dm=None, dv=None, cm='royalblue', cv='orange',
            ms=6, alpha=0.5, figsize=(9,5)):
    """Select data around planet transits.
    ----------
    df : data  frame (time [d], flux [pp1], flux_err [pp1])
    dm : model frame (time [d], flux [pp1], flux_err [pp1])
    dv : input frame (time [d], flux [pp1], flux_err [pp1])
    """
    time = df.time.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.errorbar(df.time, df.flux, yerr=df.flux_err, fmt='.k', ms=ms, alpha=alpha, zorder=1)
    if dm is not None:
        ax.plot(dm.time, dm.flux, '-', c=cm)
    if dv is not None:
        ax.plot(dv.time, dv.flux, '-', c=cv)        
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalized flux")
    ax.set_xlim(time[0], time[-1])
    plt.tight_layout()
    return fig, ax


def plot_result(df, dm, alpha=0.5, ms=3,
                cm='royalblue', lw=1.5,
                label=None, samples=[], Q0=None,
                figsize=(9,7)):
    """Plot data with best fit model and residuals.
    """
    time = df.time.to_numpy()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig)
    # Plot best-fit model light curve and data
    ax0 = fig.add_subplot(gs[0:2, 0])
    ax0.errorbar(time, df.flux, yerr=df.flux_err, fmt='ok',
                 ms=ms, alpha=alpha, zorder=1, label=label)
    # Add label
    if label is not None:
        ax0.set_label(loc='upper right')
    # Plot 95% uncertainties
    # sample = result['weighted_samples']['points']
    # quantile = int(len(sample) * (1-uncertainty))
    # for q in range(quantile):
    #     p = sample[-q-1]
    #     params = model_params()
    #     params.z     = z
    #     params.t0    = p[0]
    #     params.P     = p[1]
    #     params.i     = p[2]
    #     params.e     = p[3]
    #     params.w     = p[4]
    #     params.logM1 = p[5]
    #     params.logM2 = p[6]
    #     params.L     = p[7]
    #     params.alpha = p[8]
    #     params.vz    = 0
    #     modelfit = model(params)
    #     modelflux, _, _ = modelfit.light_curve(time)
    #     ax0.plot(time, modelflux, '-', c='orange', lw=1, alpha=0.05)
    # Plot maximum-likelihood curve
    ax0.plot(time, dm.flux, '-', c=cm, lw=lw)
    ax0.set_ylabel("Normalized flux")
    ax0.set_xlim(time[0], time[-1])
    ax0.tick_params(labelbottom=False)
    #ax0.set_xticks([])
    # Remove last major tick label
    # labels = ax0.get_yticklabels()
    # labels[0] = ""
    # ax0.=set_yticklabels(labels)
    # Plot the residuals
    if 'pred_gp_mean' in samples:
        gp_med_mean = jnp.median(samples["pred_gp_mean"], axis=0)
        gp_med_std  = jnp.median(samples["pred_gp_std"],  axis=0)
        #ax0.plot(x, gp_med_mean, color=cm)
        ax0.fill_between(time, gp_med_mean - 2*gp_med_std, gp_med_mean + 2*gp_med_std,
                         color=cm, alpha=0.3)
    if 'pred_smbhb' in samples:
        q = jnp.percentile(samples["pred_smbhb"], jnp.array([1, 50, 99]), axis=0)
        ax0.plot(time, q[1], color="orange")
        ax0.fill_between(time, q[0], q[2], color="orange", alpha=0.3)
    # Make residaul in another subplot
    residuals = df.flux.to_numpy() - dm.flux.to_numpy()
    ax1 = fig.add_subplot(gs[2, 0])
    ax1.errorbar(time, residuals, yerr=df.flux_err, fmt='ok', ms=ms, alpha=alpha, zorder=1)
    ax1.plot(time, np.zeros_like(time), '--', c=cm, lw=lw)
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Residuals")
    ax1.set_xlim(time[0], time[-1])
    # Correct labels
    for ax in [ax0, ax1]:
        ax.get_yaxis().set_label_coords(-0.09, 0.5)
    # Plot quarter marks
    if isinstance(Q0, int):
        plot_quarter_marks([ax0, ax1], time, 2, Q0=Q0)
    # Global settings
    plt.tight_layout(h_pad=0)
    fig.subplots_adjust(hspace=0.0)
    return fig, [ax0, ax1]


#--------------------------------------------------------------#
#                       ULTRANEST METHODS                      #
#--------------------------------------------------------------#

def plot_corner(result, bestfit=False, values_input=None):
    """Select data around planet transits.
    
    Parameters
    ----------
    """
    # Create corner figure
    figure = corner.corner(
        result['samples'],
        smooth=1.5,
        color='royalblue',
        labels=result['paramnames'],
        show_titles=True,
        title_kwargs={"fontsize": 18},
        quantiles=[0.16, 0.5, 0.84],
    )
    if bestfit in ['maximum', 'median', 'mean']:
        if bestfit == 'maximum':
            values_bestfit = np.array(result['maximum_likelihood']['point'])
        else:
            values_bestfit = np.array(result['posterior'][bestfit])
        corner.overplot_lines(figure,  values_bestfit, color='deeppink', lw=1)
        corner.overplot_points(figure, values_bestfit[None], marker="s", color='deeppink')
    if values_input is not None:
        if isinstance(values_input, list):
            values_input = np.array(values_input)
        corner.overplot_lines(figure,  values_input, color='orange', lw=1)    
        corner.overplot_points(figure, values_input[None], marker="s", color='orange')
    return figure


    #     gp_med_std = jnp.median(samples["pred_gp_std"], axis=0)
    #     ax.plot(x, gp_med_mean, color="royalblue")
    #     ax.fill_between(x, gp_med_mean - 2*gp_med_std, gp_med_mean + 2*gp_med_std,
    #                     color="royalblue", alpha=0.3)
    # if 'pred_smbhb' in samples:
    #     q = jnp.percentile(samples["pred_smbhb"], jnp.array([1, 50, 99]), axis=0)
    #     ax.plot(x, q[1], color="orange")
    #     ax.fill_between(x, q[0], q[2], color="orange", alpha=0.3)
    # xmin, xmax = min(x), max(x)
    # dx = (xmax - xmin) * 0.01
    # ax.set_xlim(xmin-dx, xmax+dx)

#--------------------------------------------------------------#
#                       ULTRANEST METHODS                      #
#--------------------------------------------------------------#

def plot_corner_mcmc(samples, names=None, color='royalblue', bins=30, smooth=1.2, fs=18,
                     best_params=None, best_color='k', bestfit=None,
                     true_params=None, true_color='orange'):
    """Select data around planet transits.
    """
    # Fetch parameters names
    if names is None:
        names = [k for k in samples.keys() if 'pred' not in k]
    # Fetch samples of each model parameter
    data = np.stack([samples[k] for k in names]).T
    # Create corner plot
    fig = corner.corner(
        data=data,
        labels=names,
        bins=bins,
        smooth=smooth,
        color=color,
        show_titles=True,
        title_kwargs={"fontsize": fs},
        quantiles=[0.16, 0.5, 0.84],
    )
    # Plot true parameters
    if true_params:
        if isinstance(true_params, list):
            true_params = np.array(true_params)
        corner.overplot_lines(fig,  true_params,       color=true_color, lw=1)
        corner.overplot_points(fig, true_params[None], color=true_color, marker='s')
    # Plot best-fit parameters
    if bestfit == 'mean':
        best_params = np.mean(data, axis=0)
    elif bestfit == 'median':
        best_params = np.median(data, axis=0)
    # elif bestfit == 'MLE':
    #     best_params = np.mean(data, axis=0)
    # elif bestfit == 'MAP':
    #     best_params = np.mean(data, axis=0)
    if bestfit or best_params:
        if isinstance(best_params, list):
            best_params = np.array(best_params)
        corner.overplot_lines(fig,  best_params,       color=best_color, lw=1)
        corner.overplot_points(fig, best_params[None], color=best_color, marker='s')        
    return fig


def predictive_posterior_drw(ax, samples, x):
    if 'pred_gp_mean' in samples:
        gp_med_mean = jnp.median(samples["pred_gp_mean"], axis=0)
        gp_med_std  = jnp.median(samples["pred_gp_std"], axis=0)
        ax.plot(x, gp_med_mean, color="royalblue")
        ax.fill_between(x, gp_med_mean - 2*gp_med_std, gp_med_mean + 2*gp_med_std,
                        color="royalblue", alpha=0.3)
    if 'pred_smbhb' in samples:
        q = jnp.percentile(samples["pred_smbhb"], jnp.array([1, 50, 99]), axis=0)
        ax.plot(x, q[1], color="orange")
        ax.fill_between(x, q[0], q[2], color="orange", alpha=0.3)
    xmin, xmax = min(x), max(x)
    dx = (xmax - xmin) * 0.01
    ax.set_xlim(xmin-dx, xmax+dx)    
