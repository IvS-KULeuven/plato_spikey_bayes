#!/usr/bin/env python3
"""
This python modules for model generation using JAX.
"""
import jax
import jax.numpy as jnp
from jax import lax
from astropy import constants as c

# Constants
G_CGS = c.G.cgs.value
C_CGS = c.c.cgs.value
M_SUN = c.M_sun.cgs.value
YEAR  = 31556926    # [s]
DAY   = 86400       # [s]
RAD   = jnp.pi/180  # [rad]
FLOOR = 1e-15

#--------------------------------------------------------------#
#                      INTERNAL METHODS                        #
#--------------------------------------------------------------#

def _period_observed(P, z):
    """Orbital period in observers frame [s].
    """
    return P * (1 + z)


def _semimajor_axis(P, M):
    """Semi-major axis in binary rest frame [cm].
    """
    return (G_CGS * M * P**2 / (4 * jnp.pi**2))**(1/3)     


def _mean_anomaly(t, t0, T):
    """Determine the mean anomaly, fm [rad].
    """
    return 2 * jnp.pi * (t - t0) / T


def _kepler_equation(E, e, M):
    """Represents Kepler's equation: M = E - e * sin(E)
    """
    return E - e * jnp.sin(E) - M


def _eccentric_anomaly_jax(M, e, tolerance=1e-10, maxiter=10):
    """Determine the Eccentric anomaly, E.
    
    We here solve Kepler's equation (M = E – e sin E) and use the
    Newton-Raphson method to obtain the eccentric anomaly E.
    """
    # Jax JIT is not happy with loops with unspecified max number of iterations.
    # On average 3 iterations are sufficient to converge.
    E0 = M
    init_state = (E0, jnp.zeros_like(E0, dtype=bool))

    def body_fun(_, state):
        E, converged = state
        f_E = _kepler_equation(E, e, M)
        f_prime_E = 1.0 - e * jnp.cos(E)
        E_candidate = E - f_E / f_prime_E
        # Only update entries that are not yet converged
        E_new = jnp.where(converged, E, E_candidate)
        step_err = jnp.abs(E_candidate - E)
        new_converged = jnp.logical_or(converged, step_err < tolerance)
        return (E_new, new_converged)

    E_final, conv = lax.fori_loop(0, maxiter, body_fun, init_state)
    return E_final


def _true_anomaly(fe, e):
    """Determine the true anomaly, f [rad].        
    """
    return 2 * jnp.arctan(jnp.sqrt((1 + e) / (1 - e)) * jnp.tan(fe/2))


def _radial_vector(fe, e, a):
    """Radial vector of motion, r [cm].
    """
    return a * (1 - e * jnp.cos(fe))


def _rv_semiamplitude(P, M1, M, q, a, i, e):
    """The RV semi-amplitude of secondary
    """
    K2 = (2 * jnp.pi / P) * (M1 / M) * a * jnp.sin(i) / jnp.sqrt(1 - e**2)
    K1 = q * K2
    return K1, K2


def _rv_vector(vz, K1, K2, f, e, w):
    """Projection of the velocity vector on to the line of sight.
    Equation from (Murray & Correria, 2010). Minus sign is introduced
    here as the RV is defined to be positive when object is moving away
    from the observed
    """
    vr1 = vz + K1 * (jnp.cos(w + f) + e * jnp.cos(w))
    vr2 = vz - K2 * (jnp.cos(w + f) + e * jnp.cos(w))    
    return vr1, vr2


def _xyz_orbital_plane(f, r1, a1, q, i, w, omega=jnp.pi/2):
    """Cartesian 3D position as function of time.
    """
    # Cartesian positions of primary and secondary 
    sini  = jnp.sin(i)
    cosi  = jnp.cos(i)
    #sino  = jnp.sin(omega) # 1
    #coso  = jnp.cos(omega) # 0
    sinwf = jnp.sin(w + f)
    coswf = jnp.cos(w + f)
    #x1 = r1 * (coso * coswf - sino * sinwf * cosi)
    x1 = r1 * -sinwf * cosi
    #y1 = r1 * (sino * coswf + coso * sinwf * cosi)
    y1 = r1 * coswf
    z1 = r1 * (sinwf * sini)
    x2 = -x1 / q
    y2 = -y1 / q
    z2 = -z1 / q
    return x1, y1, z1, x2, y2, z2


def _angular_separation_xy(x1, x2, y1, y2):
    """Angular separation between lens and source in cartesian coordinates, delta.
    """
    return jnp.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def _angular_einstein_radius(z1, z2, M1, M2, flip):
    """Einstein radius of primary and secondary [cm].
    """
    M_l = jnp.where(flip, M2, M1)
    D_l = jnp.where(flip, -z2, -z1)
    D_s = jnp.where(flip, -z1, -z2)
    D_rel = D_s - D_l
    return jnp.sqrt(4 * G_CGS * M_l * D_rel / C_CGS**2)


def _magnification_point(u):
    """Magnification of point source limit.
    """
    return (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))

# def soft_flip(z1, sharpness=1e2):
#    return jax.nn.sigmoid(-sharpness * z1)

#--------------------------------------------------------------#
#                      PUBLIC SMBHB CLASS                      #
#--------------------------------------------------------------#

def smbhb_jax(time, z, t0, P, i, e, w, logM1, logM2, L, alpha, vz, **kwargs):
    """Magnification of point source limit.
    """
    
    # Make sure to work with floats (to avoid int overflow)
    z     = float(z)
    t0    = t0 * YEAR
    P     = P  * YEAR
    i     = jnp.deg2rad(i)
    w     = jnp.deg2rad(w)
    M1     = 10**logM1 * M_SUN
    M2     = 10**logM2 * M_SUN
    vz    = float(vz)
    M = M1 + M2
    q = 10**(logM2 - logM1)
    
    # Orbital period in binary rest frame [s] 
    T = _period_observed(P, z)

    # Check parameters
    fm = _mean_anomaly(time*DAY, t0, T)
    fe = _eccentric_anomaly_jax(fm, e)
    #fe = eccentric_anomaly_fixed_iters(fm, e, maxiter=1)
    f  = _true_anomaly(fe, e)

    # Semi-major axis [cm]        
    a  = _semimajor_axis(P, M)
    a1 = a * M2 / M

    # Radial coordinate []
    r  = _radial_vector(fe, e, a)
    r1 = _radial_vector(fe, e, a1)
    
    # RELATIVISTIC DOPPLER BOOSTING
    
    # The RV semi-amplitude of secondary [cm/s]
    K1, K2 = _rv_semiamplitude(P, M1, M, q, a, i, e)
    
    # Projection of the velocity vector on to the line of sight [cm/s]
    vr1, vr2 = _rv_vector(vz, K1, K2, f, e, w)

    # Gamma factors for each component [cm/s]
    arg = G_CGS * M * (2/r - 1/a) / C_CGS**2
    v1_sqr = jnp.minimum(arg * (M2/M)**2, 1-FLOOR)
    v2_sqr = jnp.minimum(arg * (M1/M)**2, 1-FLOOR)
    gamma1 = 1 / jnp.sqrt(1 - v1_sqr)
    gamma2 = 1 / jnp.sqrt(1 - v2_sqr)

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
    return jnp.where(flip, F_if_secondary_lenses, F_if_primary_lenses)
