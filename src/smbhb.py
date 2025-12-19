#!/usr/bin/env python3

"""
This python module contains plot utilities used in the minimal 
PlatoSim installation and in the extra PLATOnium installation.
"""

# Built-in
import datetime

# PlatoSim standard
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d, make_interp_spline
from scipy.optimize import root_scalar
from tqdm import tqdm

# Platonium standard
from numba import jit, njit
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy import constants as c
from astropy.coordinates import SkyCoord

# UltraNest
import ultranest
import ultranest.stepsampler
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot
import corner

# PlatoSim libraries
import platosim.slurm     as sm
import platosim.utilities as ut
import platosim.plot      as pt
import platosim.noise     as ns
import platosim.starquery as sq
from platosim.lightcurve   import LightCurve
from platosim.slurm        import workerOverview
from platosim.matplotlibrc import setup_notebook
setup_notebook()

# Constants
G_CGS = c.G.cgs.value
C_CGS = c.c.cgs.value
M_SUN = c.M_sun.cgs.value
YEAR  = ut.year()
DAY   = 86400
RAD   = np.pi/180
floor = 1e-15

#--------------------------------------------------------------#
#                      INTERNAL METHODS                        #
#--------------------------------------------------------------#

@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _period_observed(P, z):
    """Orbital period in observers frame [s].
    """
    return P * (1 + z)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _semimajor_axis(P, M):
    """Semi-major axis in binary rest frame [cm].
    """
    return (G_CGS * M * P**2 / (4 * np.pi**2))**(1/3)     


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _mean_anomaly(t, t0, T):
    """Determine the mean anomaly, fm [rad].

    Parameters
    ----------
    t : Time array
    T : Orbital period (observed)
    """
    return 2 * np.pi * (t - t0) / T


# @jit(cache=True, nopython=True, fastmath=True, parallel=False)
# def _eccentric_anomaly(fm, e):
#     """Determine the Eccentric anomaly, fe.

#     We here solve Kepler's equation (fm = fe – e sin fe) and use the
#     Newton-Raphson method to obtain the eccentric anomaly fe.
#     """
#     f = lambda fe: fe - e * np.sin(fe) - fm
#     fe = root_scalar(f, x0=fm, x1=fm + 0.1, method='secant').root
#     return fe


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _kepler_equation(E, e, M):
    """Represents Kepler's equation: M = E - e * sin(E)
    """
    return E - e * np.sin(E) - M


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _eccentric_anomaly(M, e, tolerance=1e-10):
    """Determine the Eccentric anomaly, E.

    We here solve Kepler's equation (M = E – e sin E) and use the
    Newton-Raphson method to obtain the eccentric anomaly E.
    """
    # Initial guess for E. A common starting point is M itself. For better performance,
    # a more sophisticated starting point can be used (e.g., Machin's).
    E = M
    # Iterate until the solution converges
    while True:
        f_E = _kepler_equation(E, e, M)
        f_prime_E = 1 - e * np.cos(E)
        # Newton-Raphson step
        E_next = E - f_E / f_prime_E
        # Check for convergence
        if abs(E_next - E) < tolerance:
            return E_next
        # Else save next step
        E = E_next


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _true_anomaly(fe, e):
    """Determine the true anomaly, f [rad].        
    """
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(fe/2))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _radial_vector(fe, e, a):
    """Radial vector of motion, r [cm].
    """        
    return a * (1 - e * np.cos(fe))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _rv_semiamplitude(P, M1, M, q, a, i, e):
    """The RV semi-amplitude of secondary
    """
    K2 = (2 * np.pi / P) * (M1 / M) * a * np.sin(i) / np.sqrt(1 - e**2)
    K1 = q * K2
    return K1, K2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _rv_vector(vz, K1, K2, f, e, w):
    """Projection of the velocity vector on to the line of sight.

    Equation from (Murray & Correria, 2010). Minus sign is introduced
    here as the RV is defined to be positive when object is moving away
    from the observed
    """
    vr1 = vz + K1 * (np.cos(w + f) + e * np.cos(w))
    vr2 = vz - K2 * (np.cos(w + f) + e * np.cos(w))    
    return vr1, vr2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _xyz_orbital_plane(f, r1, a1, q, i, w, omega=np.pi/2):
    """Cartesian 3D position as function of time.
    """
    # Cartesian positions of primary and secondary 
    sini  = np.sin(i)
    cosi  = np.cos(i)
    sino  = np.sin(omega)
    coso  = np.cos(omega)
    sinwf = np.sin(w + f)
    coswf = np.cos(w + f)
    x1 = r1 * (coso * coswf - sino * sinwf * cosi)
    y1 = r1 * (sino * coswf + coso * sinwf * cosi)
    z1 = r1 * (sinwf * sini)
    x2 = -x1 / q
    y2 = -y1 / q
    z2 = -z1 / q
    return x1, y1, z1, x2, y2, z2


# @jit(cache=True, nopython=True, fastmath=True, parallel=False)
# def _radius_schwarzchild(M, q):
#     """Schwarzchild radius of primary and secondary [cm].
#     """
#     RS1 = 2 * C_CGS * M     / ((1 + q) * C_CGS**2)
#     RS2 = 2 * C_CGS * M * q / ((1 + q) * C_CGS**2)
#     return RS1, RS2


# def einstein_radius(self, phi1, phi2, I):
#     """Einstein radius of primary and secondary [cm].
#     """        
#     RS1, RS2 = self.RS
#     const = 2 * self.a.value * np.cos(I)
#     RE1 = np.sqrt(const * RS1.value * np.sin(phi1))
#     RE2 = np.sqrt(const * RS2.value * np.sin(phi2))        
#     return RE1, RE2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _angular_separation_xy(x1, x2, y1, y2):
    """Angular separation between lens and source in cartesian coordinates, delta.
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _angular_einstein_radius(z1, z2, M1, M2, flip):
    """Einstein radius of primary and secondary [cm].
    """
    M_l = np.full(z1.shape, M1)
    D_l = -z1
    D_s = -z2
    M_l[flip] = M2
    D_l[flip] = -z2[flip]
    D_s[flip] = -z1[flip]
    D_rel = D_s - D_l
    return np.sqrt(4 * G_CGS * M_l * D_rel / C_CGS**2)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _magnification_point(u):
    """Magnification of point source limit.
    """
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))


#--------------------------------------------------------------#
#                      PUBLIC SMBHB CLASS                      #
#--------------------------------------------------------------#

@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def time(tdur, dt, t0=0):
    """Generate time array from input parameters [d].

    Parameters
    ----------
    tdur : float [astropy.unit]
        Duration of time series.
    dt : float [astropy.unit]
        Cadence of observation.
    t0 : float [astropy.unit]
        Start of time series.
    """
    return np.arange(t0, tdur, dt)


def get_df(time, flux, flux_lens, flux_boost, flux_red=None):
    """Create a data frame with model light curve.
    """
    df = pd.DataFrame()
    df['time']       = time
    df['flux']       = flux
    df['flux_lens']  = flux_lens
    df['flux_boost'] = flux_boost
    if flux_red is not None:
        df['flux_red'] = flux_red
    return df


@njit
def getRedNoise(time, currenttime, kicktimestep, Ntime,
                timescale, varscale, noise, mu, sigma,
                rng):
        
    signal = np.zeros(Ntime)
    
    for i in range(Ntime):
        # Compute the contribution of each component separately.
        # First advance the time series right *before* the time point i,
        while ((currenttime + kicktimestep) < time[i]):
            noise = noise * (1.0 - kicktimestep/timescale) + rng.normal(mu[0], sigma[0])
            currenttime = currenttime + kicktimestep

        # Then advance the time series with a small time step right *on* time[i]
        delta = time[i] - currenttime
 
        # Correction factor to have varscale in RMS arcsec
        sigma1 = np.sqrt(delta / timescale) * varscale
        noise  = noise * (1.0 - delta/timescale) + rng.normal(mu[0], sigma1[0])
        currenttime = time[i]

        # Add the different components to the signal. 
        signal[i] = np.sum(noise)

    return signal


def modelRedNoise(time, timescale, varscale, kickscale=100, n_warmup=2000, seed=None):
    """Function to generate a red noise time series.
    
    Parameters
    ----------
    time : ndarray
        Time points: time[0..Ntime-1]
    timescale : ndarray
        Time scale tau of each red noise component: timescale[0..Ncomp-1]
    varscale : ndarray
        Variation scale of each red noise component: varscale[0..Ncomp-1]
            
    Returns
    -------
    signal : ndarray
        Signal containing all red noise components: signal[0..Ntime-1]
    """
    # Initialise random generator
    rng = ut.rng(seed=seed)

    # Correct tau
    #timescale = np.sqrt(timescale)
    
    # Shortcuts
    Ntime = len(time)
    Ncomp = len(timescale)

    # Set the kick (= excitation) timestep to be one 100th of the
    # shortest noise time scale (i.e. kick often enough).
    kicktimestep = min(timescale) / kickscale
    currenttime  = time[0] - kicktimestep
    
    # Predefine some arrays
    delta = 0.0
    noise = np.zeros(Ncomp)
    mu    = np.zeros(Ncomp)
    sigma = np.sqrt(kicktimestep / timescale) * varscale

    # Warm up the first-order autoregressive process
    for i in range(n_warmup):
        noise = noise * (1 - kicktimestep / timescale) + rng.normal(mu, sigma)

    # Start simulating the granulation time series
    signal_red = getRedNoise(time, currenttime, kicktimestep, Ntime,
                             timescale, varscale, noise, mu, sigma, rng)
    return signal_red * 1e-3 + 1
    

def modelRedNoisePSD(freq, timescale, varscale):
    """Generate a red noise model from the PSD [ppm^2/microHz].

    Compute the mean power spectral density (PSD) corresponding to the 
    red noise time series that is generated by modelRedNoise().

    Parameters
    ----------
    freq : ndarray
        frequency points of the PSD.
    timescale : float
        Also know as tau for red noise signal [inverse of freq].
     varscale : float
        Also known as sigma in red noise signal.

    Returns
    -------
    psd : ndarray
        Power spectral density (PSD) in units: [sigma]^2/[freq]
    """
    psd = np.zeros_like(freq)
    for n in range(len(timescale)):
        sigma = varscale[n]
        tau = timescale[n]
        psd += tau**2 * sigma**2 / (1 + (2 * np.pi * freq * tau)**2)
    return psd


class model_params(object):
    """Load model parameters.
    """
    def __init__(self):
        # Time array
        self.time = None   # [day]
        # Observational parameters
        self.z     = 0.    # Redshift
        self.t0    = 1.    # Time of ephemeris [yr]
        # Orbital parameters
        self.P     = 1.    # Observed period [yr]
        self.i     = 1.    # Inclination [deg]
        self.e     = 0.    # Eccentricity
        self.w     = 0.    # Argument of periapse [deg]
        # Physical parameters
        self.logM1 = 8.    # Total mass [log(M_sun)]
        self.logM2 = 7.    # Mass ratio
        self.L     = 0.1   # Luminosity ratio
        # Doppler boosting parameters
        self.alpha = 2.    # Spectral slope
        self.vz    = 0.    # Relative motion of frames [cm/s]
        # Quasar red-noise parameters
        self.tau   = None  # [day]
        self.sigma = None  # [ppm]
        self.seed  = None
        
    
class model(object):
    """Load model parameters.
    """
    def __init__(self, params):
        # Time array
        self.time = params.time
        # Observational parameters
        self.t0 = params.t0
        self.z  = params.z
        # Orbital parameters
        self.P = params.P
        self.i = params.i
        self.e = params.e
        self.w = params.w
        # Physical parameters
        self.logM1 = params.logM1
        self.logM2 = params.logM2
        self.L     = params.L        
        # Doppler boosting parameters
        self.alpha = params.alpha
        self.vz    = params.vz
        # Quasar red-noise parameters
        self.tau   = params.tau
        self.sigma = params.sigma
        self.seed  = params.seed
    
    def light_curve(self, time, df=False):
        """Generate light curve from model parameters. 
        """
        flux, flux_boost, flux_lens = smbhb(
            # Time array
            time,
            # Observational parameters
            self.z,
            self.t0,
            # Orbital parameters
            self.P,
            self.i,
            self.e,
            self.w,
            # Physical parameters
            self.logM1,
            self.logM2,
            self.L,
            # Doppler boosting parameters
            self.alpha,
            self.vz,
            # Quasar red-noise parameters
            self.tau,
            self.sigma,
            self.seed
        )
        # Add red noise model if requested
        if self.tau is not None:
            #-------------------------------------------------------------------
            # from astroML.time_series import generate_damped_RW
            # flux_red = generate_damped_RW(time, tau=self.tau, SFinf=self.sigma,
            #                               z=self.z, random_state=self.seed) + 1
            #-------------------------------------------------------------------
            tau   = np.array([self.tau])
            sigma = np.array([self.sigma])
            flux_red = modelRedNoise(time, tau, sigma, seed=self.seed)
            flux += (flux_red - 1)
            if df:
                return get_df(time, flux, flux_lens, flux_boost, flux_red)
            else:
                return flux, flux_lens, flux_boost, flux_red
        else:
            if df:
                get_df(time, flux, flux_lens, flux_boost)
            else:
                return flux, flux_lens, flux_boost

    
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def smbhb(time, z, t0, P, i, e, w, logM1, logM2, L, alpha, vz, tau, sigma, seed):

    # Make sure to work with floats (to avoid int overflow)
    z     = float(z)
    t0    = t0 * YEAR
    P     = P  * YEAR
    i     = np.deg2rad(i)
    e     = float(e)
    w     = np.deg2rad(w)
    M1    = 10**logM1 * M_SUN
    M2    = 10**logM2 * M_SUN
    L     = float(L)
    alpha = float(alpha)
    vz    = float(vz)
    # if tau is not None:
    #     tau = float(tau)
    # if sigma is not None:
    #     sigma = float(sigma)
    # if seed is not None:
    #     seed = int(seed) 

    # Total mass and mass fraction
    M = M1 + M2
    q = M2 / M1 

    # Orbital period in observers frame [s]
    T = _period_observed(P, z)

    # Check parameters
    fm = _mean_anomaly(time*DAY, t0, T)
    fe = np.array([_eccentric_anomaly(m, e) for m in fm])
    f  = _true_anomaly(fe, e)

    # Semi-major axis [cm]        
    a  = _semimajor_axis(P, M)
    a1 = a * M2 / M

    # Radial coordinate [cm]
    r  = _radial_vector(fe, e, a)
    r1 = _radial_vector(fe, e, a1)
    
    # RELATIVISTIC DOPPLER BOOSTING
    
    # The RV semi-amplitude of secondary [cm/s]
    K1, K2 = _rv_semiamplitude(P, M1, M, q, a, i, e)

    # Projection of the velocity vector on to the line of sight [cm/s]
    vr1, vr2 = _rv_vector(vz, K1, K2, f, e, w)

    # Gamma factors for each component [cm/s]
    arg = G_CGS * M * (2/r - 1/a) / C_CGS**2
    v1_sqr = np.minimum(arg * (M2/M)**2, 1-floor)
    v2_sqr = np.minimum(arg * (M1/M)**2, 1-floor)
    gamma1 = 1 / np.sqrt(1 - v1_sqr)
    gamma2 = 1 / np.sqrt(1 - v2_sqr)

    # Relativistic doppler boosting [pp1]
    D1 = 1 / (gamma1 * (1 - vr1/C_CGS))**(3 - alpha)
    D2 = 1 / (gamma2 * (1 - vr2/C_CGS))**(3 - alpha)
    D  = (1 - L) * D1 + L * D2
    
    # GRAVITATIONAL SELF-LENSING
    
    # Find cartesian position vectors
    x1, y1, z1, x2, y2, z2 = _xyz_orbital_plane(f, r1, a1, q, i, w)

    # Switch to select secondary as lens (or primary as source)
    flip = (z1 < 0)

    # Point-source magnification
    delta = _angular_separation_xy(x1, x2, y1, y2)
    theta = _angular_einstein_radius(z1, z2, M1, M2, flip)
    u     = delta / (theta + floor)
    M_ps  = _magnification_point(u)

    # COMBINE MODEL COMPONENTS

    x = flip
    flux    = (1 - L) * D1             + L * D2 * M_ps
    flux[x] = (1 - L) * D1[x]* M_ps[x] + L * D2[x]

    # Return: flux_total, flux_boost, flux_lens
    return flux, D, M_ps


#--------------------------------------------------------------#
#                     PUBLIC ULTRANET CLASS                    #
#--------------------------------------------------------------#

class model_priors(object):
    """Initialise model priors.
    """
    def __init__(self):
        # Observational parameters
        self.z     = [0, 3]
        self.t0    = [0, 5]
        # Orbital parameters
        self.P     = [0, 5]
        self.i     = [0, 90]
        self.e     = [0, 1]
        self.w     = [0, 360]
        # Physical parameters
        self.logM1 = [6, 11]
        self.logM2 = [6, 11]
        self.L     = [0, 1]
        # Doppler boosting parameters
        self.alpha = [-4, 4]
        self.vz    = [0, 1]
        # Quasar red-noise parameters
        self.tau   = None
        self.sigma = None


def run_ultranest(df, priors, path, nsteps=1000, live_points=400):

     # Convert observation to numpy arrays
    time = df.time.to_numpy()
    flux = df.flux.to_numpy()
    flux_err = df.flux_err.to_numpy()

    # Check if prior is range or value
    params_names = []
    priors_range = []
    # names_params = ['z', 't0', 'P', 'i', 'e', 'w',
    #                'logM', 'q', 'L', 'alpha', 'vz',
    #                'tau', 'sigma']
    # names_priors = []
    # for name,prior in zip(names_params, names_priors):                  

    if type(priors.z) == list:
        params_names.append('z')
        priors_range.append(priors.z)
    elif type(priors.z) in [int, float]:
        z = float(priors.z)
    
    if type(priors.t0) == list:
        params_names.append('t0')
        priors_range.append(priors.t0)
    elif type(priors.t0) in [int, float]:
        t0 = float(priors.t0)

    if type(priors.P) == list:
        params_names.append('P')
        priors_range.append(priors.P)
    elif type(priors.P) in [int, float]:
        P = float(priors.P)

    if type(priors.i) == list:
        params_names.append('i')
        priors_range.append(priors.i)
    elif type(priors.i) in [int, float]:
        i = float(priors.i)

    if type(priors.e) == list:
        params_names.append('e')
        priors_range.append(priors.e)
    elif type(priors.e) in [int, float]:
        e = float(priors.e)

    if type(priors.w) == list:
        params_names.append('w')
        priors_range.append(priors.w)
    elif type(priors.w) in [int, float]:
        w = float(priors.w)

    if type(priors.logM1) == list:
        params_names.append('logM1')
        priors_range.append(priors.logM1)
    elif type(priors.logM1) in [int, float]:
        logM1 = float(priors.logM1)

    if type(priors.logM2) == list:
        params_names.append('logM2')
        priors_range.append(priors.logM2)
    elif type(priors.logM2) in [int, float]:
        logM2 = float(priors.logM2)
        
    if type(priors.L) == list:
        params_names.append('L')
        priors_range.append(priors.L)
    elif type(priors.L) in [int, float]:
        L = float(priors.L)

    if type(priors.alpha) == list:
        params_names.append('alpha')
        priors_range.append(priors.alpha)
    elif type(priors.alpha) in [int, float]:
        params_names.append('alpha')
        alpha = float(priors.alpha)
        priors_range.append(alpha)
        #normal = scipy.stats.norm(alpha, 0.5)
        
    if type(priors.vz) == list:
        params_names.append('vz')
        priors_range.append(priors.vz)
    elif type(priors.vz) in [int, float]:
        vz = float(priors.vz)

    if type(priors.tau) == list:
        params_names.append('tau')
        priors_range.append(priors.tau)
    elif type(priors.tau) in [int, float]:
        tau = float(priors.tau)

    if type(priors.sigma) == list:
        params_names.append('sigma')
        priors_range.append(priors.sigma)
    elif type(priors.sigma) in [int, float]:
        sigma = float(priors.sigma)

    # We do not use wrapping so set all elements to False
    n = len(params_names)
    wrapped_params = np.zeros(n).astype(bool)

    # print(normal.ppf(alpha))
    # exit()
    
    # Define a few function needed for UltraNest

    # def transform_normal(quantile):
    #     return normal.ppf(quantile)
    
    def prior_transform(cube):
        """Prior transformation hypercube.
        """
        p = cube.copy()
        x = priors_range 
        for i in range(n):
            # if i == if type(normal) == scipy.stats._distn_infrastructure.rv_continuous_frozen:
            #     p[i] = normal.ppf(cube[i])
            # else:
            p[i] = cube[i] * (x[i][1] - x[i][0]) + x[i][0]
        return p

    def log_likelihood(p):
        """Simple log-likehood using trial parameters.
        """        
        flux_trial, _, _ = smbhb(
            time  = time,
            z     = z,
            t0    = p[0],
            P     = p[1],
            i     = p[2],
            e     = p[3],
            w     = p[4],
            logM1 = p[5],
            logM2 = p[6],
            L     = p[7],
            alpha = p[8],
            vz    = vz,
            tau   = None,
            sigma = None,
            seed  = None
            # Red noise parameters
            #         tau   = 50
            #         sigma = 300
            #         seed  = 123456789
            #         cache  = cache  # Can't use cache because free LDs
        )
        return -0.5 * np.nansum(((flux_trial - flux) / flux_err)**2)

    # Initialise sampler
    sampler = ReactiveNestedSampler(
        params_names,
        log_likelihood, 
        prior_transform,
        wrapped_params = wrapped_params,
        log_dir        = path,
        resume         = 'overwrite',
#        vectorized     = True
    )

    # Set number of step and 
    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
        nsteps          = nsteps,
        max_nsteps      = 5000,
        adaptive_nsteps = 'move-distance',
    )

    # Run nested sampling
    tic  = datetime.datetime.now()
    result = sampler.run(min_num_live_points=live_points)
    toc  = datetime.datetime.now()
    print(f'Execution time: {toc-tic} [h:mm:ss]')
    
    # Always show the distributions
    sampler.print_results()

    # Always show the
    # filename = path / "info/results.json"
    # with open(filename, 'r') as g:
    #     logz = json.load(g)
    # print("logz:", results["logz"])
    
    # Return result and sampler
    return result, sampler


#--------------------------------------------------------------#
#                      PUBLIC PLOT METHODS                     #
#--------------------------------------------------------------#

def plot_corner(result, bestfit=False, values_input=None):
    """Select data around planet transits.
    
    Parameters
    ----------
    """    
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
        corner.overplot_lines(figure,  values_input, color='orange', lw=1)    
        corner.overplot_points(figure, values_input[None], marker="s", color='orange')
    return figure


def plot_result(df, result, z,
                tdur=3, dt=0.1,
                alpha=0.2,
                show_quarters=True,
                figsize=(9,7)):

    # Fetch names of model parameters
    #parameters = result['paramnames']
    
    # Maximum likelihood parameters
    mll_t0    = result["maximum_likelihood"]["point"][0]
    mll_P     = result["maximum_likelihood"]["point"][1]
    mll_i     = result["maximum_likelihood"]["point"][2]
    mll_e     = result["maximum_likelihood"]["point"][3]
    mll_w     = result["maximum_likelihood"]["point"][4]
    mll_logM  = result["maximum_likelihood"]["point"][5]
    mll_q     = result["maximum_likelihood"]["point"][6]
    mll_L     = result["maximum_likelihood"]["point"][7]
    mll_alpha = result["maximum_likelihood"]["point"][8]

    # Initialise model
    params = model_params()
    params.z     = z
    params.t0    = mll_t0
    params.P     = mll_P
    params.logM  = mll_logM
    params.q     = mll_q
    params.i     = mll_i
    params.e     = mll_e
    params.w     = mll_w
    params.L     = mll_L
    params.alpha = mll_alpha
    params.vz    = 0

    # Generate model from best-fit parameters
    time = df.time.to_numpy()
    modelfit = model(params)
    modelflux, _, _ = modelfit.light_curve(time)
    dm = pd.DataFrame({'time': time, 'flux': modelflux})

    
    # Plot light curve
    residuals = df.flux - dm.flux
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0:2, 0])
    ax0.errorbar(df.time, df.flux, yerr=df.flux_err, fmt='.k', alpha=alpha, zorder=1)
    ax0.plot(dm.time, dm.flux, '-', c='deeppink')
    ax0.set_ylabel("Normalized flux")
    ax0.set_xlim(time[0], time[-1])
    
    ax1 = fig.add_subplot(gs[2, 0])
    ax1.errorbar(df.time, residuals, yerr=df.flux_err, fmt='.k', alpha=alpha, zorder=1)
    ax1.plot(dm.time, np.zeros_like(dm.time), '--', c='orange')
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Residuals")
    ax1.set_xlim(time[0], time[-1])

    plt.tight_layout()
    return fig, [ax0, ax1]


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


def plot_lc(df, dm=None, dv=None, ms=10, alpha=0.5, figsize=(9,5)):
    """Select data around planet transits.
    
    Parameters
    ----------
    df : light curve with columns: time [d], flux [pp1]
    dv : input model with columns: time [d], flux [pp1]
    """
    time = df.time.to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.errorbar(df.time, df.flux, yerr=df.flux_err, fmt='.k', alpha=alpha, zorder=1)
    if dm is not None:
        ax.plot(dm.time, dm.flux, '-', c='royalblue')
    if dv is not None:
        ax.plot(dv.time, dv.flux, '-', c='orange')        
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalized flux")
    ax.set_xlim(time[0], time[-1])
    plt.tight_layout()
    return fig, ax


    # # Check DRW model
    # Q = self.quasar_variability(verbose=verbose, plot=plot, plot_psd=False)
    # if quasar:
    #     try:
    #         Q = self.quasar_variability(verbose=verbose, plot=plot, plot_psd=False)
    #     except:
    #         Q = np.ones_like(time)

    # # Check boosting model
    # try:
    #     D, D1, D2 = self.doppler_boosting(verbose=verbose, plot=plot)
    # except:
    #     D = D1 = D2 = np.ones_like(self.t)

    # # Check lensing model
    # try:
    #     M_ps, m_ps = self.gravitational_lensing(verbose=verbose, plot=plot)
    # except:
    #     M_ps = m_ps = np.ones_like(self.t)
    #     flux    = (1 - self.L) * D1 * M_ps + self.L * D2 * M_ps

    # # Interpolate back to original time grid and add DRW model
    # D_interp    = make_interp_spline(self.t.to('s').value, D,    k=3)
    # m_ps_interp = make_interp_spline(self.t.to('s').value, m_ps, k=3)
    # flux_interp = make_interp_spline(self.t.to('s').value, flux, k=3)
    # D    = D_interp(self.time.to('s').value)
    # m_ps = m_ps_interp(self.time.to('s').value)
    # flux = flux_interp(self.time.to('s').value)
    # flux += (Q - 1)

    # # Find boosting+lensing model
    # print(len(D1), len(M_ps))
    # print(len(self.time), len(self.t))
    # x = self.flip
    # flux    = (1 - self.L) * D1             + self.L * D2 * M_ps
    # flux[x] = (1 - self.L) * D1[x]* M_ps[x] + self.L * D2[x]


    # if plot:
    #     fig = plt.figure(figsize = (9, 5))
    #     plt.plot(time, Q,     color='tomato',    label="DRW",      lw=0.8)
    #     plt.plot(time, D,     color='orange',    label="Beaming",  lw=1.8)
    #     plt.plot(time, m_ps,  color='royalblue', label="Lensing",  lw=1.8)
    #     plt.plot(time, flux,  color='k',         label="Combined", lw=0.8)
    #     plt.xlabel(r"Time [day]")
    #     plt.ylabel(r"Relative flux")
    #     plt.xlim(0, time[-1])
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    #     if ofile_fig:
    #         fig.savefig(ofile_fig, bbox_inches='tight', dpi=300)

    # return flux, Q, D, m_ps

    
#--------------------------------------------------------------#
#                        PUBLIC METHODS                        #
#--------------------------------------------------------------#

def quasar_variability(self, verbose=False, plot=False, plot_psd=False):
    """Initialise ANG intrinsic variability.

    This model uses a damped random walk (i.e. red noise) descroption
    to compute the quasar variability.

    Parameters

    ----------
    tau : ndarray
        Time scale tau of each red noise component [s]
    sigma : ndarray 
        Variation scale of each red noise component [ppm]
    verbose : bool
        Option to print info to bash
    plot : bool
        Show normalised light curve [pp1]
    plot_psd : bool
        Show Power Spectral Density (PSD) plot.

    Returns
    -------
    Q : Signal containing all red noise components [pp1]
    """
    tau   = np.array([self.tau.to('d').value])
    sigma = np.array([self.sigma])

    # Show parameters to screen
    if verbose:
        ut.errorcode('message', '\nDRW parameters:')
        print(f'Damping timesclae,           tau : {tau[0]:.1f} d')
        print(f'Std of variability,        sigma : {sigma[0]:.1f} ppm')

    # Run model (NOTE we use original time array here)
    time  = self.time.to('d').value
    self.Q = ns.modelRedNoise(time, tau, sigma, seed=self.seed) * 1e-6 + 1
    print(plot)
    if plot:
        fig = plt.figure(figsize=(9,5))
        plt.plot(time, self.Q, c='tomato')
        plt.xlabel(r"Time [day]")
        plt.ylabel(r"Relative flux")
        plt.xlim(0, time[-1])
        plt.tight_layout()
        plt.show()

    if plot_psd:
        dt = np.diff(time)[0]
        Nfreq = time.shape[0]
        freq  = np.arange(float(Nfreq)) / (Nfreq-1) / (2*dt)
        PSD   = ns.modelRedNoisePSD(freq, tau, sigma)
        # Show PSD plot
        fig = plt.figure(figsize=(9,5))
        for i in range(len(tau)):
            plt.loglog(freq, PSD, c='tomato', lw=2)
        plt.xlabel(r"Frequency [c/d]")
        plt.ylabel(r"PSD [ppm$^2$ d$^2$]")
        plt.xlim(np.min(freq), np.max(freq))
        plt.tight_layout()

    return self.Q


def doppler_boosting(self, verbose=False, plot=False):
    """Model relativistic Doppler boosting.

    This function initialise the Doppler boosting model of a
    two-body gravitationally bound system.

    Parameters
    ----------
    alpha : float 
        Spectral index of both mini-discs
    vz : float
        Barycentric velocity of system [cm/s]

    Return
    ------
    Relative flux time series of doppler boosting signal.
    """

    # The RV semi-amplitude of secondary [cm/s]
    K1,K2 = self._rv_semiamplitude(self.P, self.M1, self.M, self.q, self.a, self.i, self.e)

    # Show parameters to screen
    if verbose:
        ut.errorcode('message', '\nDoppler boosting parameters:')
        print(f'Spectral index minidiscs,  alpha : {self.alpha:.2f}')
        print(f'RV semi-amplitude of primary, K1 : {K1.to("km/s"):.2f}')
        print(f'RV semi-amplitude of second., K2 : {K2.to("km/s"):.2f}')

    # Check parameters
    try:
        E = self.E
        f = self.f
        r = self.r            
    except AttributeError:
        #E = np.array([eccentric_anomaly(m, self.e) for m in self.fm])
        E = np.array([self._eccentric_anomaly(m, self.e) for m in self.fm])
        f = self._true_anomaly(E, self.e)
        r = self._radial_vector(E, self.e, self.a)

    # Projection of the velocity vector on to the line of sight [cm/s]
    vr1, vr2 = self._rv_vector(self.vz, K1, K2, f, self.e, self.w)

    # Relativistic doppler boosting [pp1]
    arg1 = (self.M2 / self.M)**2 * c.G.cgs * self.M * (2/r - 1/self.a) / c.c.cgs**2
    arg2 = (self.M1 / self.M)**2 * c.G.cgs * self.M * (2/r - 1/self.a) / c.c.cgs**2
    v1_sqr = np.minimum(arg1.value, 1-floor)
    v2_sqr = np.minimum(arg2.value, 1-floor)
    gamma1 = 1 / np.sqrt(1 - v1_sqr)
    gamma2 = 1 / np.sqrt(1 - v2_sqr)
    self.d1 = 1 / (gamma1 * (1 - vr1/c.c.cgs))**(3 - self.alpha)
    self.d2 = 1 / (gamma2 * (1 - vr2/c.c.cgs))**(3 - self.alpha)
    self.d  = (1 - self.L) * self.d1 + self.L * self.d2

    if plot:
        t = self.t.to('d').value
        fig = plt.figure(figsize = (9, 5))
        plt.plot(t, self.d1, ':',  c='orange', label=r"$D_1$")
        plt.plot(t, self.d2, '-.', c='orange', label=r"$D_2$")
        plt.plot(t, self.d,  '-',  c='orange', label=r"$D$")
        plt.xlabel(r"Time [day]")
        plt.ylabel(r"Relative flux")
        plt.xlim(0, t[-1])
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Interpolate back to original time grid
    D_interp  = make_interp_spline(self.t.cgs.value, self.d,  k=3)        
    D1_interp = make_interp_spline(self.t.cgs.value, self.d1, k=3)
    D2_interp = make_interp_spline(self.t.cgs.value, self.d2, k=3)
    self.D  = D_interp( self.time.cgs.value)
    self.D1 = D1_interp(self.time.cgs.value)
    self.D2 = D2_interp(self.time.cgs.value)        

    return self.D, self.D1, self.D2


def gravitational_lensing(self, verbose=False, plot=False):
    """Model gravitational self-lensing.

    This function calculates the magnification of a SMBHB binary pair
    accretion discs during their orbital phase. The model return both
    the magnification in the point source (PS) and finite source (FS)
    limit.

    Parameters
    ----------
    J : float, astropy.unit [deg, rad]
    wvl : float, astropy.unit [nm, cm, m]
    u_max : Maximum number of Einstein radii to compute finite lensing.
    u_num : Number of grid points in u plane.
    v_num : Number of grid points in v plane.
    verbose : bool
    plot : bool

    Returns
    -------
    Magnification of primary and secondary: M1_ps [pp1], M2_ps [pp1]
    """
    #J   = J.to('rad')
    #wvl = wvl.cgs

    # Print to bash
    if verbose:
        RS1, RS2 = self._radius_schwarzchild(self.M, self.q)
        ut.errorcode('message', '\nSelf-lensing parameters:')
        #print(f'Inclination of mini-disc,    J   : {J.to("deg"):.1f}')            
        #print(f'Inclination of mini-disc,    wvl : {wvl.to("nm"):.0f}')
        print(f'Schwarchild radius primary,  Rs1 : {RS1.to("R_sun"):.2f}')
        print(f'Schwarchild radius second.,  Rs2 : {RS2.to("R_sun"):.2f}')

    # Find anomalies
    try:
        E = self.E
        f = self.f
    except AttributeError:
        E = np.array([self._eccentric_anomaly(m, self.e) for m in self.fm])
        f = self._true_anomaly(E, self.e)

    # Find cartesian position vectors
    x1, y1, z1, x2, y2, z2 = self._xyz_orbital_plane(self.a1.value, self.q, self.i,
                                                     self.e, self.w, E, f)

    # Switch to select secondary as lens (or primary as source)
    self.flip = (z1 < 0)

    # Point-source magnification
    delta = self._angular_separation_xy(x1, x2, y1, y2)
    theta = self._angular_einstein_radius(z1, z2, self.M1.value, self.M2.value)
    u = delta / (theta + floor)
    self.M_ps = self._magnification_point(u)

    # Finite-source magnification
    # delta_u = u_max / u_num        
    # delta_v = 2 * np.pi / v_num
    # u_array = np.linspace(0, u_max, u_num)
    # v_array = np.linspace(0, 2*np.pi, v_num)
    # u_grid, v_grid = np.meshgrid(u_array, v_array)
    # u0 = u
    # v0 = np.arctan(self.sini * np.tan(self.fm))
    # r0 = np.array([self._radius_disc(u_grid, v_grid, u, v, theta_E, J.value)
    #                for u, v, theta_E in zip(u0, v0, theta)])
    # flux = np.array([self._flux_disc(r, z, wvl.value, self.a, self.M, self.q)
    #                  for r,z in zip(r0, z1)])
    # M_fs = self._magnification_finite(flux, u_grid, delta_u, delta_v)

    # Compute point-source magnification with luminosity ratio
    x = self.flip
    D1 = D2 = np.ones_like(u)
    self.m_ps    = (1 - self.L) * D1                   + self.L * D2 * self.M_ps
    self.m_ps[x] = (1 - self.L) * D1[x] * self.M_ps[x] + self.L * D2[x]        
    # self.m_fs    = (1 - self.L) * D1                   + self.L * D2 * self.M_fs
    # self.m_fs[x] = (1 - self.L) * D1[x] * self.M_fs[x] + self.L * D2[x]

    if plot:
        t = self.t.to('d').value
        plt.figure(figsize=(9,5))
        plt.plot(t, self.M_ps, '-.', c='royalblue', label=r"$\mathcal{M}^{\rm PS}$")
        plt.plot(t, self.m_ps, '-',  c='royalblue', label=r"$\mathcal{m}^{\rm PS}$")
        # plt.plot(t, self.M_fs, '-.', c='b', label=r"$\mathcal{M}^{\rm FS}$")
        # plt.plot(t, self.m_fs, '-',  c='b', label=r"$\mathcal{m}^{\rm FS}$")
        plt.xlabel(r"Time [day]")
        plt.ylabel(r"Relative flux")
        plt.xlim(0, t[-1])
        plt.legend()
        plt.tight_layout()
        plt.show()

    return self.M_ps, self.m_ps


#---------------------------------------------------------------
#  FUNCTIONS FOR NOTEBOOK: 1. Star catalogues
#---------------------------------------------------------------

def fetch_gaia_info(df, NED=False):
    """Fetch Gaia info for each source in data frame.
    Use NASA/IPAC Extragalactic Database (NED).
    """
    for i in range(df.shape[0]):
        di = df.reset_index(drop=True).loc[i]
        if NED:
            ra, dec = di.RA, di.Dec
        else:
            ra, dec = di.ra, di.dec
        dx = sq.gaiaQueryCone(ra, dec, radius=0.001, mag_max=21)
        if dx.shape[0] > 1:
            dx = dx[dx.dis == 0]
        if i == 0:
            dq = dx
        else:
            dq = pd.concat((dq, dx))
            
    # Alter data frame
    dq = dq.reset_index(drop=True)
    dq = dq.drop(columns='dis')
    if NED:
        dq.insert(0, 'source', df['Object Name'])
        dq['z'] = df['Redshift (z)']
        
    return dq


def plot_aitoff(df_agn, df_all=False, df_lop=False, df_best=False, NED=False):
    """Fetch Gaia info for each source in data frame.
    Use NASA/IPAC Extragalactic Database (NED).
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
