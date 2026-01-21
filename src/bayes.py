#!/usr/bin/env python3
"""
Python modules for Bayesian inferences.
"""
# Built-in functions
import datetime
from functools import partial

# Repository dependencies
import numpy as np
import pandas as pd
# MCMC inference
import jax
from jax import numpy as jnp
import numpyro
import numpyro.distributions as dist
# Nested sampling
import ultranest
import ultranest.stepsampler
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot
# Gaussian process
from tinygp import GaussianProcess, kernels

# Internal dependencies
import smbhb as smbhb
from smbhb_jax import smbhb_jax

#--------------------------------------------------------------#
#                         NUMPYRO METHODS                        #
#--------------------------------------------------------------#

def make_smbhb_model(*, priors, build_mean=None):
    """
        Returns a NumPyro model function:
        model(x, yerr, y=None, fixed=None, x_interp=None)
    """
    def model(x, yerr, y=None, x_interp=None):
        params = {}
        for name, val in priors.items():
            if isinstance(val, dist.Distribution):
                params[name] = numpyro.sample(name, val)
            else:
                params[name] = val

        if build_mean is not None:
            mean = build_mean(params)
        elif 'mean' in params:
            mean = params['mean']
        else:
            raise ValueError("Provide either a mean function or set builder_mean='mean'")
        if callable(mean):
            numpyro.deterministic("pred_smbhb", jax.vmap(mean)(x_interp))
    return model


def make_tinygp_model(*, priors, build_mean=None, build_kernel=None):
    """Function to generate NumPyro Gaussian process model.
 
    Returns a NumPyro model function:
x    model(x, yerr, y=None, fixed=None, x_interp=None)
    """
    def model(x, yerr, y=None, x_interp=None):
        params = {}
        for name, val in priors.items():
            if isinstance(val, dist.Distribution):
                params[name] = numpyro.sample(name, val)
            else:
                params[name] = val

        if build_mean is not None:
            mean = build_mean(params)
        elif 'mean' in params:
            mean = params['mean']
        else:
            raise ValueError("Provide either a mean function or set builder_mean='mean'")
        if build_kernel is not None:
            kernel = build_kernel(params)
            gp = GaussianProcess(kernel, x, diag=jnp.square(yerr), mean=mean)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
            if (y is not None) and (x_interp is not None):
                cond = gp.condition(y, x_interp).gp
                numpyro.deterministic("pred_gp_mean", cond.loc)
                numpyro.deterministic("pred_gp_std", jnp.sqrt(cond.variance))
        else:
            if not callable(mean):
                raise ValueError('You have to provide a mean function or a kernel function')
            numpyro.sample('obs', dist.Normal(jax.vmap(mean)(x), yerr).to_event(1), obs=y)
        if callable(mean):
            numpyro.deterministic("pred_smbhb", jax.vmap(mean)(x_interp))
    return model


def drw_kernel(p):
    return (p["sigma"]**2) * kernels.quasisep.Exp(scale=p["tau"])


def smbhb_mean_builder(p):
    return partial(smbhb_jax, **p)


def get_drw_lc(time, samples):
    """Fetch best-fit DRW model light curve.
    """
    if isinstance(time, pd.DataFrame):
        time = time.to_numpy()
    gp_med_mean = jnp.median(samples["pred_gp_mean"], axis=0)
    gp_med_std  = jnp.median(samples["pred_gp_std"],  axis=0)
    return pd.DataFrame({'time': time, 'flux': gp_med_mean, 'flux_err': gp_med_std})


#--------------------------------------------------------------#
#                        ULTRANEST METHODS                     #
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

class model_priors_q(object):
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
        self.logM  = [5, 11]
        self.q     = [0, 1]
        self.L     = [0, 1]
        # Doppler boosting parameters
        self.alpha = [-4, 4]
        self.vz    = [0, 1]
        # Quasar red-noise parameters
        self.tau   = None
        self.sigma = None

        

# def run_ultranest(df, priors, path, nsteps=1000, live_points=400):
#     """Run UlstraNest using input priors.
#     """
#     # Convert observation to numpy arrays
#     time = df.time.to_numpy()
#     flux = df.flux.to_numpy()
#     flux_err = df.flux_err.to_numpy()

#     # Check if prior is range or value
#     params_names = []
#     priors_range = []
#     # names_params = ['z', 't0', 'P', 'i', 'e', 'w',
#     #                'logM', 'q', 'L', 'alpha', 'vz',
#     #                'tau', 'sigma']
#     # names_priors = []
#     # for name,prior in zip(names_params, names_priors):                  

#     if type(priors.z) == list:
#         params_names.append('z')
#         priors_range.append(priors.z)
#     elif type(priors.z) in [int, float]:
#         z = float(priors.z)
    
#     if type(priors.t0) == list:
#         params_names.append('t0')
#         priors_range.append(priors.t0)
#     elif type(priors.t0) in [int, float]:
#         t0 = float(priors.t0)

#     if type(priors.P) == list:
#         params_names.append('P')
#         priors_range.append(priors.P)
#     elif type(priors.P) in [int, float]:
#         P = float(priors.P)

#     if type(priors.i) == list:
#         params_names.append('i')
#         priors_range.append(priors.i)
#     elif type(priors.i) in [int, float]:
#         i = float(priors.i)

#     if type(priors.e) == list:
#         params_names.append('e')
#         priors_range.append(priors.e)
#     elif type(priors.e) in [int, float]:
#         e = float(priors.e)

#     if type(priors.w) == list:
#         params_names.append('w')
#         priors_range.append(priors.w)
#     elif type(priors.w) in [int, float]:
#         w = float(priors.w)

#     if type(priors.logM1) == list:
#         params_names.append('logM1')
#         priors_range.append(priors.logM1)
#     elif type(priors.logM1) in [int, float]:
#         logM1 = float(priors.logM1)

#     if type(priors.logM2) == list:
#         params_names.append('logM2')
#         priors_range.append(priors.logM2)
#     elif type(priors.logM2) in [int, float]:
#         logM2 = float(priors.logM2)
        
#     if type(priors.L) == list:
#         params_names.append('L')
#         priors_range.append(priors.L)
#     elif type(priors.L) in [int, float]:
#         L = float(priors.L)

#     if type(priors.alpha) == list:
#         params_names.append('alpha')
#         priors_range.append(priors.alpha)
#     elif type(priors.alpha) in [int, float]:
#         params_names.append('alpha')
#         alpha = float(priors.alpha)
#         priors_range.append(alpha)
#         #normal = scipy.stats.norm(alpha, 0.5)
        
#     if type(priors.vz) == list:
#         params_names.append('vz')
#         priors_range.append(priors.vz)
#     elif type(priors.vz) in [int, float]:
#         vz = float(priors.vz)

#     if type(priors.tau) == list:
#         params_names.append('tau')
#         priors_range.append(priors.tau)
#     elif type(priors.tau) in [int, float]:
#         tau = float(priors.tau)

#     if type(priors.sigma) == list:
#         params_names.append('sigma')
#         priors_range.append(priors.sigma)
#     elif type(priors.sigma) in [int, float]:
#         sigma = float(priors.sigma)

#     # We do not use wrapping so set all elements to False
#     n = len(params_names)
#     wrapped_params = np.zeros(n).astype(bool)

#     # print(normal.ppf(alpha))
#     # exit()
    
#     # Define a few function needed for UltraNest

#     # def transform_normal(quantile):
#     #     return normal.ppf(quantile)
    
#     def prior_transform(cube):
#         """Prior transformation hypercube.
#         """
#         p = cube.copy()
#         x = priors_range 
#         for i in range(n):
#             # if i == if type(normal) == scipy.stats._distn_infrastructure.rv_continuous_frozen:
#             #     p[i] = normal.ppf(cube[i])
#             # else:
#             p[i] = cube[i] * (x[i][1] - x[i][0]) + x[i][0]
#         return p

#     def log_likelihood(p):
#         """Simple log-likehood using trial parameters.
#         """        
#         flux_trial, _, _ = smbhb(
#             time  = time,
#             z     = z,
#             t0    = p[0],
#             P     = p[1],
#             i     = p[2],
#             e     = p[3],
#             w     = p[4],
#             logM1 = p[5],
#             logM2 = p[6],
#             L     = p[7],
#             alpha = p[8],
#             vz    = vz,
#             tau   = None,
#             sigma = None,
#             seed  = None
#             # Red noise parameters
#             #         tau   = 50
#             #         sigma = 300
#             #         seed  = 123456789
#             #         cache  = cache  # Can't use cache because free LDs
#         )
#         return -0.5 * np.nansum(((flux_trial - flux) / flux_err)**2)

#     # Initialise sampler
#     sampler = ReactiveNestedSampler(
#         params_names,
#         log_likelihood, 
#         prior_transform,
#         wrapped_params = wrapped_params,
#         log_dir        = path,
#         resume         = 'overwrite',
# #        vectorized     = True
#     )

#     # Set number of step and 
#     sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
#         nsteps          = nsteps,
#         max_nsteps      = 5000,
#         adaptive_nsteps = 'move-distance',
#     )

#     # Run nested sampling
#     tic  = datetime.datetime.now()
#     result = sampler.run(min_num_live_points=live_points)
#     toc  = datetime.datetime.now()
#     print(f'Execution time: {toc-tic} [h:mm:ss]')
    
#     # Always show the distributions
#     sampler.print_results()

#     # Always show the
#     # filename = path / "info/results.json"
#     # with open(filename, 'r') as g:
#     #     logz = json.load(g)
#     # print("logz:", results["logz"])
    
#     # Return result and sampler
#     return result, sampler


def run_ultranest(df, params, priors, path):

    import numpy as np
    import datetime
    # Fetch numpy arrays
    time = df.time.to_numpy()
    flux = df.flux.to_numpy()
    flux_err = df.flux_err.to_numpy()
    # Define model parameters
    params_input   = [params.t0, params.P, params.i, params.e, params.w,
                      params.logM1, params.logM2, params.L, params.alpha]
    params_names   = ['t0', 'P', 'i', 'e', 'w', 'logM1', 'logM2', 'L', 'alpha']
    params_wrapped = [False] * len(params_names)
    # Define prior transform
    def prior_transform(cube):
        p = cube.copy()
        p[0] = cube[0] * (priors.t0[1]    - priors.t0[0])    + priors.t0[0]
        p[1] = cube[1] * (priors.P[1]     - priors.P[0])     + priors.P[0] 
        p[2] = cube[2] * (priors.i[1]     - priors.i[0])     + priors.i[0] 
        p[3] = cube[3] * (priors.e[1]     - priors.e[0])     + priors.e[0] 
        p[4] = cube[4] * (priors.w[1]     - priors.w[0])     + priors.w[0] 
        p[5] = cube[5] * (priors.logM1[1] - priors.logM1[0]) + priors.logM1[0] 
        p[6] = cube[6] * (priors.logM2[1] - priors.logM2[0]) + priors.logM2[0] 
        p[7] = cube[7] * (priors.L[1]     - priors.L[0])     + priors.L[0] 
        p[8] = cube[8] * (priors.alpha[1] - priors.alpha[0]) + priors.alpha[0] 
        return p
    # Define log-likelihood function
    def log_likelihood(p):       
        flux_model = smbhb.smbhb(
            time  = time,
            z     = params.z,
            t0    = p[0],
            P     = p[1],
            i     = p[2],
            e     = p[3],
            w     = p[4],
            logM1 = p[5],
            logM2 = p[6],
            L     = p[7],
            alpha = p[8],
            vz    = params.vz,
            tau   = None,
            sigma = None,
            seed  = None
        )
        return -0.5 * np.nansum(((flux_model - flux) / flux_err)**2)
    # Initialise sampler
    sampler = ReactiveNestedSampler(
        param_names    = params_names,
        loglike        = log_likelihood, 
        transform      = prior_transform,
        wrapped_params = params_wrapped,
        log_dir        = path,
        resume         = 'overwrite',
    )
    # Initialise step sampler
    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
        nsteps          = 1000,
        max_nsteps      = 1000,
        adaptive_nsteps = 'move-distance',
    )
    # Run nested sampling
    tic  = datetime.datetime.now()
    result = sampler.run(min_num_live_points=400)
    toc  = datetime.datetime.now()
    print(f'Execution time: {toc-tic} [h:mm:ss]')
    sampler.print_results()

    return result, sampler


def run_ultranest_q(df, params, priors, path):

    import numpy as np
    import datetime
    # Fetch numpy arrays
    time = df.time.to_numpy()
    flux = df.flux.to_numpy()
    flux_err = df.flux_err.to_numpy()
    # Define model parameters
    params_input   = [params.t0, params.P, params.i, params.e, params.w,
                      params.logM, params.q, params.L, params.alpha]
    params_names   = ['t0', 'P', 'i', 'e', 'w', 'logM', 'q', 'L', 'alpha']
    params_wrapped = [False] * len(params_names)
    # Define prior transform
    def prior_transform(cube):
        p = cube.copy()
        p[0] = cube[0] * (priors.t0[1]    - priors.t0[0])    + priors.t0[0]
        p[1] = cube[1] * (priors.P[1]     - priors.P[0])     + priors.P[0] 
        p[2] = cube[2] * (priors.i[1]     - priors.i[0])     + priors.i[0] 
        p[3] = cube[3] * (priors.e[1]     - priors.e[0])     + priors.e[0] 
        p[4] = cube[4] * (priors.w[1]     - priors.w[0])     + priors.w[0] 
        p[5] = cube[5] * (priors.logM[1]  - priors.logM[0])  + priors.logM[0] 
        p[6] = cube[6] * (priors.q[1]     - priors.q[0])     + priors.q[0] 
        p[7] = cube[7] * (priors.L[1]     - priors.L[0])     + priors.L[0] 
        p[8] = cube[8] * (priors.alpha[1] - priors.alpha[0]) + priors.alpha[0] 
        return p
    # Define log-likelihood function
    def log_likelihood(p):       
        flux_model = smbhb.smbhb_q(
            time  = time,
            z     = params.z,
            t0    = p[0],
            P     = p[1],
            i     = p[2],
            e     = p[3],
            w     = p[4],
            logM  = p[5],
            q     = p[6],
            L     = p[7],
            alpha = p[8],
            vz    = params.vz,
            tau   = None,
            sigma = None,
            seed  = None
        )
        return -0.5 * np.nansum(((flux_model - flux) / flux_err)**2)
    # Initialise sampler
    sampler = ReactiveNestedSampler(
        params_names,
        log_likelihood, 
        prior_transform,
        wrapped_params = params_wrapped,
        log_dir        = path,
        resume         = 'overwrite',
    )
    # Initialise step sampler
    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
        nsteps          = 1000,
        max_nsteps      = 1000,
        adaptive_nsteps = 'move-distance',
    )
    # Run nested sampling
    tic  = datetime.datetime.now()
    result = sampler.run(min_num_live_points=400)
    toc  = datetime.datetime.now()
    print(f'Execution time: {toc-tic} [h:mm:ss]')
    sampler.print_results()

    return result, sampler



def bestfit_model(time, params, result, likelihood='maximum_likelihood', value='point'):
    """Fetch the best-fit model light curve.
    """
    l = likelihood
    v = value
    p = result["paramnames"]
    # Fetch parameters
    if 'z' in p: z = result[l][v][p.index('z')]
    else: z = params.z
    if 't0' in p: t0 = result[l][v][p.index('t0')]
    else: t0 = params.t0
    if 'P' in p: P = result[l][v][p.index('P')]
    else: P = params.P
    if 'i' in p: i = result[l][v][p.index('i')]
    else: i = params.i
    if 'e' in p: e = result[l][v][p.index('e')]
    else: e = params.e
    if 'w' in p: w = result[l][v][p.index('w')]
    else: w = params.w
    if 'logM1' in p: logM1 = result[l][v][p.index('logM1')]
    else: logM1 = params.logM1
    if 'logM2' in p: logM2 = result[l][v][p.index('logM2')]
    else: logM2 = params.logM2
    if 'L' in p: L = result[l][v][p.index('L')]
    else: L = params.L
    if 'alpha' in p: alpha = result[l][v][p.index('alpha')]
    else: alpha = params.alpha
    if 'vz' in p: vz = result[l][v][p.index('vz')]
    else: vz = params.vz
    if 'tau' in p: tau = result[l][v][p.index('tau')]
    else: tau = params.tau
    if 'sigma' in p: sigma = result[l][v][p.index('sigma')]
    else: sigma = params.sigma
    if 'seed' in p: seed = result[l][v][p.index('seed')]
    else: seed = params.seed    
    # Initialise model
    params_model = smbhb.model_params()
    params_model.z     = float(z)
    params_model.t0    = float(t0)
    params_model.P     = float(P)
    params_model.i     = float(i)
    params_model.e     = float(e)
    params_model.w     = float(w)
    params_model.logM1 = float(logM1)
    params_model.logM2 = float(logM2)
    params_model.L     = float(L)
    params_model.alpha = float(alpha)
    params_model.vz    = 0.0
    params_model.tau   = float(tau)  # Needs to be non-zero
    params_model.sigma = 0.0
    params_model.seed  = seed
    # Evaluate model for each point in time grid    
    model_bestfit = smbhb.model(params)
    return model_bestfit.light_curve(time, df=True)


def bestfit_model_q(time, params, result, likelihood='maximum_likelihood', value='point'):
    """Fetch the best-fit model light curve.
    """
    l = likelihood
    v = value
    p = result["paramnames"]
    # Fetch parameters
    if 'z' in p: z = result[l][v][p.index('z')]
    else: z = params.z
    if 't0' in p: t0 = result[l][v][p.index('t0')]
    else: t0 = params.t0
    if 'P' in p: P = result[l][v][p.index('P')]
    else: P = params.P
    if 'i' in p: i = result[l][v][p.index('i')]
    else: i = params.i
    if 'e' in p: e = result[l][v][p.index('e')]
    else: e = params.e
    if 'w' in p: w = result[l][v][p.index('w')]
    else: w = params.w
    if 'logM' in p: logM = result[l][v][p.index('logM')]
    else: logM = params.logM
    if 'q' in p: q = result[l][v][p.index('q')]
    else: q = params.q
    if 'L' in p: L = result[l][v][p.index('L')]
    else: L = params.L
    if 'alpha' in p: alpha = result[l][v][p.index('alpha')]
    else: alpha = params.alpha
    if 'vz' in p: vz = result[l][v][p.index('vz')]
    else: vz = params.vz
    if 'tau' in p: tau = result[l][v][p.index('tau')]
    else: tau = params.tau
    if 'sigma' in p: sigma = result[l][v][p.index('sigma')]
    else: sigma = params.sigma
    if 'seed' in p: seed = result[l][v][p.index('seed')]
    else: seed = params.seed
    # Initialise model
    params_model = smbhb.model_params_q()
    params_model.z     = float(z)
    params_model.t0    = float(t0)
    params_model.P     = float(P)
    params_model.i     = float(i)
    params_model.e     = float(e)
    params_model.w     = float(w)
    params_model.logM  = float(logM)
    params_model.q     = float(q)
    params_model.L     = float(L)
    params_model.alpha = float(alpha)
    params_model.vz    = 0.0
    params_model.tau   = float(tau) # Needs to be non-zero
    params_model.sigma = 0.0
    params_model.seed  = seed
    # Evaluate model for each point in time grid    
    model_bestfit = smbhb.model_q(params)
    return model_bestfit.light_curve(time, df=True)
