#!/usr/bin/env python3
"""
Python modules for Bayesian inferences.
"""
# Built-in functions
import datetime
from functools import partial

# Repository dependencies
import scipy
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
# JAX
import jax
from jax import random
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)
# JAXNS
from jaxns import summary
# NymPyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.util import log_density
from numpyro.contrib.nested_sampling import NestedSampler
# Gaussian process
from tinygp import GaussianProcess, kernels
# Nested sampling
import ultranest
import ultranest.stepsampler
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot

# Internal dependencies
import smbhb as smbhb
import plots as pt
from smbhb_jax import smbhb_jax, smbhb_jax_q

#--------------------------------------------------------------#
#                       INTERNAL METHODS                       #
#--------------------------------------------------------------#

def _get_numpy_arrays(df):
    """Fetch a list of model parameter names.
    """
    time = df.time.to_numpy()
    flux = df.flux.to_numpy()
    ferr = df.flux_err.to_numpy()
    return time, flux, ferr


def _get_jax_arrays(df):
    """Fetch a list of model parameter names.
    """    
    time = jnp.asarray(df.time)
    flux = jnp.asarray(df.flux)
    ferr = jnp.asarray(df.flux_err)
    return time, flux, ferr


def _get_param_names(priors):
    """Fetch a list of model parameter names.
    """
    names = []
    for n,v in zip(priors.keys(), priors.values()):
        if not isinstance(v, float):
            names.append(n)
    return names


def _get_params_dict(params):
    """Transform parameter class instance to dictionary.
    """
    params_dict = {
        'z'    : params.z,         
        't0'   : params.t0, 
        'P'    : params.P, 
        'i'    : params.i, 
        'e'    : params.e, 
        'w'    : params.w, 
        'logM1': params.logM1, 
        'logM2': params.logM2, 
        'L'    : params.L, 
        'alpha': params.alpha,
        'vz'   : params.vz,
        'tau'  : params.tau,
        'sigma': params.sigma,
        'seed' : params.seed         
    }
    return params_dict


def _log_posterior(df, model, params):
    """Fetch log posterior density.
    """
    lp, _ = log_density(
        model,
        model_args=(_get_jax_arrays(df)),
        model_kwargs={},
        params=params
    )
    return lp


def _get_mle(df, priors, build_kernel, build_mean, samples, names):
    """Fetch Maximum Likelihood Estimate (MPLE).
    """
    posterior = [{k: samples[k][i].item() for k in names} for i in range(len(names))]
    model  = make_tinygp_model(
        priors=priors,
        build_kernel=build_kernel,
        build_mean=build_mean
    )
    logps = [_log_posterior(df, model, p) for p in posterior]
    imax  = int(jnp.argmax(jnp.array(logps)))
    return posterior[imax]


def _get_mode(sample):
    """Fetch mode of posterior.
    """    
    return scipy.stats.mode(sample)[0]


def _get_percentile(sample, pt):
    """Fetch percentile of posterior.
    """    
    return np.percentile(sample, pt, axis=0)


def _get_posterior(samples, names, pt_low=16, pt_upp=84, latex=False):
    """Fetch posteriors from samples.
    """
    posterior = {
        'map': np.array([_get_mode(samples[n]) for n in names], dtype=float).tolist(),
        'mean': np.array([np.mean(samples[n]) for n in names], dtype=float).tolist(),
        'median': np.array([np.median(samples[n]) for n in names]).tolist(),
        'std': np.array([np.std(samples[n]) for n in names]).tolist(),
        'err_low': np.array([_get_percentile(samples[n], pt_low) -
                             _get_percentile(samples[n], 50) for n in names]).tolist(),
        'err_upp': np.array([_get_percentile(samples[n], pt_upp) -
                             _get_percentile(samples[n], 50) for n in names]).tolist(),
    }
    if latex:
        print(posterior["mean"][0]); exit()
        for n,i in zip(names, len(names)):
            print(n,': pmx','{',
                  f'{posterior["mean"][i]:.4f}','}{',
                  f'{posterior["err_low"][i]:.4f}','}{',
                  f'{posterior["err_upp"][i]:.4f}','}')
    
    return posterior


def _save_result(params, names, values, priors, samples, values_mle, ofile=None,
                 logl=0.0, logz=0.0, logz_err=0.0):
    """Fetch and save result dictionary.
    """
    posterior = _get_posterior(samples, names)
    result = {
        # Injected parameters
        'params'   : params,
        'names'    : names,
        'values'   : values,
        # Bayesian inferences
        'logz': logz,
        'logz_err': logz_err,
        'priors'   : priors,
        'samples'  : samples,
        'posterior': {
            'map'    : posterior['map'],
            'mean'   : posterior['mean'],
            'median' : posterior['median'],
            'std'    : posterior['std'], 
            'err_low': posterior['err_low'],
            'err_upp': posterior['err_upp'],
        },
        'likelihood': {
            'logl': logl,
            'mle' : values_mle,
        },       
    }
    if ofile:
        #ofile.parent.mkdir(parents=True, exist_ok=True)
        np.save(ofile, result)
    return result

#--------------------------------------------------------------#
#                  NUMPYRO AND JAXNS METHODS                   #
#--------------------------------------------------------------#

def model_lightcurve_drw(time, result):
    """Model light curve of DRW from params.
    """
    time_int = jnp.linspace(jnp.amin(time), jnp.amax(time), len(time))
    flux_int = jnp.mean(result['samples']['pred_gp_mean'], axis=0)
    # flux = jnp.mean(result['samples']['pred_gp_mean'], axis=0)
    # time_int = jnp.linspace(jnp.amin(time), jnp.amax(time), len(time))
    # interp = scipy.interpolate.make_interp_spline(time, flux, k=3)
    # flux_int = interp(time_int)
    return pd.DataFrame({'time': time_int, 'flux': flux_int})


def model_priors_dict(params, logMq=False):
    """Model priors of Spikey in dict.
    """
    if logMq:
        priors = {
            'z'    : params.z,
            'vz'   : params.vz,
            't0'   : dist.Uniform(0, 3),
            'P'    : dist.Uniform(0, 5),
            'i'    : dist.Uniform(0, 90),
            'e'    : dist.Uniform(0, 1),
            'w'    : dist.Uniform(0, 360),
            'logM' : dist.Uniform(5, 11),
            'q'    : dist.Uniform(0, 1),
            'alpha': dist.Uniform(-4, 4),
            'L'    : dist.Uniform(0, 1),
        }
    else:
        priors = {
            'z'    : params.z,
            'vz'   : params.vz,            
            't0'   : dist.Uniform(0, 3),
            'P'    : dist.Uniform(0, 5),
            'i'    : dist.Uniform(0, 90),
            'e'    : dist.Uniform(0, 1),
            'w'    : dist.Uniform(0, 360),
            'logM1': dist.Uniform(5, 11),
            'logM2': dist.Uniform(5, 11),
            'alpha': dist.Uniform(-4, 4),            
            'L'    : dist.Uniform(0, 1),
        }        
    return priors


def make_smbhb_model(*, priors, build_mean=None):
    """Function to generate a Numpyro function for DM model. 
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
        if callable(mean) and x_interp is not None:
            numpyro.deterministic("pred_smbhb", jax.vmap(mean)(x_interp))
    return model


def make_tinygp_model(*, priors, build_mean=None, build_kernel=None):
    """Function to generate NumPyro Gaussian process model.
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
        # Case where a mean model is parsed
        if build_mean is not None:
            mean = build_mean(params)
        elif 'mean' in params:
            mean = params['mean']
        else:
            raise ValueError("Provide either a mean function or set builder_mean='mean'")
        # Case to handle DRW modelling with GP
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
        #
        if callable(mean) and x_interp is not None:
            numpyro.deterministic("pred_smbhb", jax.vmap(mean)(x_interp))
    return model


def drop_pred_params(results):
    """Drop the quantiles of the predictive posterior.
    """
    filtered_samples = {
        k: v for k, v in results.samples.items()
        if not k.startswith("pred_")
    }
    return results._replace(samples=filtered_samples)


def drw_kernel(p):
    return (p["sigma"]**2) * kernels.quasisep.Exp(scale=p["tau"])


def smbhb_mean_builder(p, q=False):
    return partial(smbhb_jax, **p)


def smbhb_mean_builder_q(p):
    return jax.vmap(partial(smbhb_jax_q, **p))


def get_drw_lc(time, samples):
    """Fetch best-fit DRW model light curve.
    """
    if isinstance(time, pd.DataFrame):
        time = time.to_numpy()
    gp_med_mean = jnp.median(samples["pred_gp_mean"], axis=0)
    gp_med_std  = jnp.median(samples["pred_gp_std"],  axis=0)
    return pd.DataFrame({'time': time, 'flux': gp_med_mean, 'flux_err': gp_med_std})


def run_numpyro(df, params, priors, build_mean,
                build_kernel=None,
                target_accept_prob=0.9,
                dense_mass=True,
                max_tree_depth=5,
                num_warmup=1000,
                num_samples=10_000,
                num_chains=4,
                progress_bar=True,
                jit_model_args=True,
                ofile=None):
    """Function to run NumPyro modelling.
    """    
    # Initialise NUTS kernel
    kernel = NUTS(
        make_tinygp_model(
            priors=priors,
            build_kernel=build_kernel,
            build_mean=build_mean
        ), 
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth
    )
    # Intialise MCMC class
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        jit_model_args=jit_model_args
    )
    # RNG key to reproduce results
    rng_key = jax.random.PRNGKey(params.seed)
    # Fetch numpy arrays
    time, flux, flux_err = _get_numpy_arrays(df)
    # Run MCMC analysis
    mcmc.run(rng_key, x=time, y=flux, yerr=flux_err, x_interp=time)
    samples = mcmc.get_samples()
    mcmc.print_summary()
    # Get dict of input parameters and lists model parameters
    params = _get_params_dict(params)
    names  = _get_param_names(priors)
    values = [params[n] for n in names]
    # Get maximum likelihood estimates (MLE)
    values_mle = _get_mle(df, priors, build_kernel, build_mean, samples, names)
    # Return and save result dictionary
    return _save_result(params, names, values, priors, samples, values_mle, ofile)


def run_jaxns(df, params, priors, build_mean,
              build_kernel=None,
              num_live_points=2000,
              num_samples=10_000,
              max_samples=100_000,
              init_efficiency_threshold=0.1,
              ofile=None):
    """Function to run JAXNS modelling.
    """    
    # Initialise JAXNS
    ns = NestedSampler(
        make_tinygp_model(
            priors=priors, 
            build_kernel=build_kernel, 
            build_mean=build_mean
        ), 
        constructor_kwargs={
            'num_live_points': num_live_points, 
            'max_samples': max_samples, 
            'init_efficiency_threshold': init_efficiency_threshold,
            'gradient_guided': False, 
            'parameter_estimation': False, 
            'difficult_model': True,
            'verbose': False 
        }
    )
    # Random number generator
    rng_key1, rng_key2 = random.split(random.PRNGKey(params.seed))
    # Fetch JAX arrays
    time, flux, flux_err = _get_jax_arrays(df)
    # Run JAXNS analysis
    time_interp = jnp.linspace(jnp.amin(time), jnp.amax(time), len(time))
    ns.run(rng_key1, x=time, y=flux, yerr=flux_err, x_interp=time_interp)
    summary(drop_pred_params(ns._results))
    # Select a sub-sample for plot
    samples = ns.get_samples(rng_key2, num_samples=num_samples)
    # Get dict of input parameters and lists model parameters
    params = _get_params_dict(params)
    if build_mean is None:
        params['mean'] = 1.0
    names  = _get_param_names(priors)
    values = [params[n] for n in names]
    # Get maximum likelihood estimates (MLE)
    logl = jnp.amax(ns._results.log_L_samples).item()
    logz = ns._results.log_Z_mean.item()
    logz_err = ns._results.log_Z_uncert.item()
    values_mle = _get_mle(df, priors, build_kernel, build_mean, samples, names)
    # Return and save result dictionary
    return _save_result(params, names, values, priors, samples, values_mle, ofile,
                        logl, logz, logz_err)


def get_posterior_clusters(df, result,
                           param_cluster='w',
                           n_components=2,
                           plot=True):
    """Function to get cluster structues in posteriors.
    """
    # Fetch result entries
    params  = result['params']
    names   = result['names']
    values  = result['values']
    priors  = result['priors']
    samples = result['samples']
    # Seperate cluster with Gaussian mixture model    
    cluster_model = GaussianMixture(n_components=n_components)
    response = cluster_model.fit_predict(samples[param_cluster].reshape(-1, 1))
    pt.plot_clusters(samples, cluster_model, response, param_cluster)
    # Fetch each cluster
    clusters = []
    for c in range(cluster_model.n_components):
        # Fetch cluster sample
        sample_cluster = {k: v[response==c] for k,v in samples.items()}
        # Get maximum likelihood estimates (MLE)
        build_kernel = None
        build_mean   = smbhb_mean_builder
        values_mle = _get_mle(df, priors, build_kernel, build_mean, samples, names)
        # Append each cluster to result
        clusters.append(
            _save_result(params, names, values, priors, sample_cluster, values_mle)
        )
    return clusters
        
#--------------------------------------------------------------#
#                      ULTRANEST METHODS                       #
#--------------------------------------------------------------#

class model_priors(object):
    """Initialise model priors.
    """
    def __init__(self):
        # Observational parameters
        self.z     = [0, 3]
        self.t0    = [0, 3]
        # Orbital parameters
        self.P     = [0, 3]
        self.i     = [0, 90]
        self.e     = [0, 1]
        self.w     = [0, 180]
        # Physical parameters
        self.logM1 = [5, 11]
        self.logM2 = [5, 11]
        self.L     = [0, 1]
        # Doppler boosting parameters
        self.alpha = [-4, 4]
        self.vz    = [0, 1]
        # Quasar red-noise parameters
        self.tau   = None
        self.sigma = None

        
def run_ultranest(df, params, priors, path, live_points=400):
    """Run nested sampling with UltraNest.
    """
    # Fetch numpy arrays
    time = df.time.to_numpy()
    flux = df.flux.to_numpy()
    flux_err = df.flux_err.to_numpy()
    # Define model parameters
    params_input   = [params.t0, params.P, params.i, params.e, params.w,
                      params.logM1, params.logM2, params.L, params.alpha]
    params_names   = ['t0', 'P', 'i', 'e', 'w', 'logM1', 'logM2', 'L', 'alpha']
    params_wrapped = [False] * len(params_names)
    params_wrapped[4] = True # For w 
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
    # Control logging
    import sys
    import logging
    logger = logging.getLogger("ultranest")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('[ultranest] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    # Run nested sampling
    tic  = datetime.datetime.now()
    result = sampler.run(min_num_live_points=live_points)
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


#------------------------------------------------------------------------------------

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

def run_ultranest_q(df, params, priors, odir,
                    nsteps=1000,
                    max_nsteps=1000,
                    min_num_live_points=400,
                    report_time=False):

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
        log_dir        = odir,
        resume         = 'overwrite',
    )
    # Initialise step sampler
    sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
        nsteps          = nsteps,
        max_nsteps      = max_nsteps,
        adaptive_nsteps = 'move-distance',
    )
    # Run nested sampling
    if report_time:
        tic  = datetime.datetime.now()
        result = sampler.run(min_num_live_points=min_num_live_points)
        toc  = datetime.datetime.now()
        print(f'Execution time: {toc-tic} [h:mm:ss]')
    else:
        result = sampler.run(min_num_live_points=min_num_live_points)
    # Print and return results
    sampler.print_results()
    return result, sampler


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
