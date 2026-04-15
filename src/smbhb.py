#!/usr/bin/env python3
"""
This python modules for model generation using Numba JIT.
"""
# Built-in
import datetime

# uv dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline
from scipy.optimize import root_scalar
from astropy import units as u
from astropy import constants as c
from astropy.coordinates import SkyCoord
from numba import jit, njit
from tqdm import tqdm

# Internal dependencies
import utils as ut
import plots as pt

# Constants
G_CGS = c.G.cgs.value
C_CGS = c.c.cgs.value
M_SUN = c.M_sun.cgs.value
YEAR  = 31556926   # [s]
DAY   = 86400      # [s]
RAD   = np.pi/180  # [rad]
FLOOR = 1e-15

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
    """
    return 2 * np.pi * (t - t0) / T


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
    arg = np.cos(w + f) + e * np.cos(w)
    vr1 = vz + K1 * arg
    vr2 = vz - K2 * arg
    return vr1, vr2


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def _xyz_orbital_plane(f, r1, a1, q, i, w, Omega=np.pi/2):
    """Cartesian 3D position as function of time.
    """
    # Cartesian positions of primary and secondary 
    sini  = np.sin(i)
    cosi  = np.cos(i)
    sino  = np.sin(Omega)
    coso  = np.cos(Omega)
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



@njit
def _get_red_noise(time, currenttime, kicktimestep, Ntime,
                   timescale, varscale, noise, mu, sigma, rng):
    """Magnification of point source limit.
    """
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


def _model_red_noise(time, timescale, varscale, kickscale=100, n_warmup=2000, seed=None):
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
    signal_red = _get_red_noise(time, currenttime, kicktimestep, Ntime,
                                timescale, varscale, noise, mu, sigma, rng)
    return signal_red * 1e-3 + 1


#--------------------------------------------------------------#
#                        PUBLIC METHODS                        #
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
    """Create a data frame with each light curve model component.
    """
    df = pd.DataFrame()
    df['time']       = time
    df['flux']       = flux
    df['flux_lens']  = flux_lens
    df['flux_boost'] = flux_boost
    if flux_red is not None:
        df['flux_red'] = flux_red
    return df


def model_lightcurve(time, values, df=True):
    """Save UltraNest result to a json file.
    """        
    if isinstance(values, dict):
        params = model_params()
        params.t0    = values['t0']
        params.P     = values['P']
        params.i     = values['i']
        params.e     = values['e']
        params.w     = values['w']
        params.logM1 = values['logM1']
        params.logM2 = values['logM2']
        params.L     = values['L']
        params.alpha = values['alpha']
        params.sigma = 0
    model_lc = model(params)
    return model_lc.light_curve(time, df=df)


#-----------------------------------------------------------------

class model_params(object):
    """Load model parameters. 
    Default parameters are for Spikey (Hu+2020, Table 1).
    """
    def __init__(self):
        # Time array
        self.time = None   # [day]
        # Relativistic model
        self.z     = 0.918  # Redshift
        self.t0    = 1.050  # Time of ephemeris [yr]
        self.P     = 1.144  # Observed period [yr]
        self.i     = 81.95  # Inclination [deg]
        self.e     = 0.524  # Eccentricity
        self.w     = 84.63  # Argument of periapse [deg]
        self.logM1 = 7.4    # Mass primary [log(M_sun)]
        self.logM2 = 6.7    # Mass secondary [log(M_sun)]
        self.L     = 0.89   # Luminosity ratio
        self.alpha = 2.09   # Spectral slope
        self.vz    = 0.     # Relative motion of frames [c]
        # Damped Random Walk
        self.tau   = 31.    # [day] (10 mmag = 9.25 ppt -> ut.mmag2ppt(10))
        self.sigma = 9.25   # [ppt]
        self.seed  = 12345  # Default seed used for paper
        
    
class model(object):
    """Load model parameters.
    """
    def __init__(self, params):
        self.time  = params.time
        self.t0    = params.t0
        self.z     = params.z
        self.P     = params.P
        self.i     = params.i
        self.e     = params.e
        self.w     = params.w
        self.logM1 = params.logM1
        self.logM2 = params.logM2
        self.L     = params.L        
        self.alpha = params.alpha
        self.vz    = params.vz
        self.tau   = params.tau
        self.sigma = params.sigma
        self.seed  = params.seed
    
    def light_curve(self, time, df=False):
        """Generate light curve from model parameters. 
        """
        flux, flux_boost, flux_lens = smbhb(
            time,
            self.z,
            self.t0,
            self.P,
            self.i,
            self.e,
            self.w,
            self.logM1,
            self.logM2,
            self.L,
            self.alpha,
            self.vz,
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
            flux_red = _model_red_noise(time, tau, sigma, seed=self.seed)
            flux += (flux_red - 1)
            if df:
                return get_df(time, flux, flux_lens, flux_boost, flux_red)
            else:
                return flux, flux_lens, flux_boost, flux_red
        else:
            if df:
                return get_df(time, flux, flux_lens, flux_boost)
            else:
                return flux, flux_lens, flux_boost

    
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def smbhb(time, z, t0, P, i, e, w, logM1, logM2, L, alpha, vz, tau, sigma, seed):
    """Magnification of point source limit.
    """
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
    v1_sqr = np.minimum(arg * (M2/M)**2, 1-FLOOR)
    v2_sqr = np.minimum(arg * (M1/M)**2, 1-FLOOR)
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
    u     = delta / (theta + FLOOR)
    M_ps  = _magnification_point(u)

    # FINAL LIGHT CURVE

    F_if_secondary_lenses = (1 - L) * D1 * M_ps + L * D2
    F_if_primary_lenses   = (1 - L) * D1        + L * D2 * M_ps
    flux = np.where(flip, F_if_secondary_lenses, F_if_primary_lenses)
    return flux, D, M_ps


#-----------------------------------------------------------------

class model_params_q(object):
    """Load model parameters.
    """
    def __init__(self):
        # Time array
        self.time = None   # [day]
        # Relativistic model
        self.z     = 0.918  # Redshift
        self.t0    = 1.050  # Time of ephemeris [yr]
        self.P     = 1.144  # Observed period [yr]
        self.i     = 81.95  # Inclination [deg]
        self.e     = 0.524  # Eccentricity
        self.w     = 84.63  # Argument of periapse [deg]
        self.logM  = 7.479  # Total mass [log(M_sun)]
        self.q     = 0.1995 # Mass ratio
        self.L     = 0.89   # Luminosity ratio
        self.alpha = 2.09   # Spectral slope
        self.vz    = 0.     # Relative motion of frames [cm/s]
        # Quasar red-noise
        self.tau   = 31.    # [day] (10 mmag = 9.25 ppt -> ut.mmag2ppt(10))
        self.sigma = 9.25   # [ppm]
        self.seed  = 12345  # Default seed used through out paper
    
class model_q(object):
    """Load model parameters.
    """
    def __init__(self, params):
        self.time  = params.time
        self.t0    = params.t0
        self.z     = params.z
        self.P     = params.P
        self.i     = params.i
        self.e     = params.e
        self.w     = params.w
        self.logM  = params.logM
        self.q     = params.q
        self.L     = params.L        
        self.alpha = params.alpha
        self.vz    = params.vz
        self.tau   = params.tau
        self.sigma = params.sigma
        self.seed  = params.seed
    
    def light_curve(self, time, df=False):
        """Generate light curve from model parameters. 
        """
        flux, flux_boost, flux_lens = smbhb_q(
            time,
            self.z,
            self.t0,
            self.P,
            self.i,
            self.e,
            self.w,
            self.logM,
            self.q,
            self.L,
            self.alpha,
            self.vz,
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
            flux_red = _model_red_noise(time, tau, sigma, seed=self.seed)
            flux += (flux_red - 1)
            if df:
                return get_df(time, flux, flux_lens, flux_boost, flux_red)
            else:
                return flux, flux_lens, flux_boost, flux_red
        else:
            if df:
                return get_df(time, flux, flux_lens, flux_boost)
            else:
                return flux, flux_lens, flux_boost

    
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def smbhb_q(time, z, t0, P, i, e, w, logM, q, L, alpha, vz, tau, sigma, seed):
    """Magnification of point source limit.
    """
    # Make sure to work with floats (to avoid int overflow)
    z     = float(z)
    t0    = t0 * YEAR
    P     = P  * YEAR
    i     = np.deg2rad(i)
    e     = float(e)
    w     = np.deg2rad(w)
    M     = 10**logM * M_SUN
    q     = float(q)
    L     = float(L)
    alpha = float(alpha)
    vz    = float(vz)

    # Binary masses
    M1 = M / (1 + q)
    M2 = M - M1
    
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
    v1_sqr = np.minimum(arg * (M2/M)**2, 1-FLOOR)
    v2_sqr = np.minimum(arg * (M1/M)**2, 1-FLOOR)
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
    u     = delta / (theta + FLOOR)
    M_ps  = _magnification_point(u)

    # FINAL LIGHT CURVE

    F_if_secondary_lenses = (1 - L) * D1 * M_ps + L * D2
    F_if_primary_lenses   = (1 - L) * D1        + L * D2 * M_ps
    flux = np.where(flip, F_if_secondary_lenses, F_if_primary_lenses)
    return flux, D, M_ps
