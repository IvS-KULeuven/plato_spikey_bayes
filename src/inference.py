from functools import partial
import jax
from jax import numpy as jnp
import numpyro
import numpyro.distributions as dist
from tinygp import GaussianProcess, kernels

from smbhb_jax import smbhb_two_masses

def box_constraint(eta, min_val, max_val):
    return min_val + (max_val-min_val)*jax.nn.sigmoid(eta)

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
            raise ValueError("You have to either give a mean function or a parameter called mean")
        if callable(mean):
            numpyro.deterministic("pred_smbhb", jax.vmap(mean)(x_interp))
    return model



def make_tinygp_model(*, priors, build_mean=None, build_kernel=None):
    """
        Returns a NumPyro model function:
        model(x, yerr, y=None, fixed=None, x_interp=None)
    """
    def model(x, yerr, y=None, x_interp=None):
        params = {}
        for name, val in priors.items():
            if isinstance(val, dist.Distribution):
                params[name] = numpyro.sample(name, val)
            elif isinstance(val, tuple):
                params[name] = numpyro.sample(name, val[0])
                repara_name = name.split('eta_')[1]
                params[repara_name] = numpyro.deterministic(repara_name, box_constraint(params[name], val[1], val[2]))
            else:
                params[name] = val

        if build_mean is not None:
            mean = build_mean(params)
        elif 'mean' in params:
            mean = params['mean']
        else:
            raise ValueError("You have to either give a mean function or a parameter called mean")
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
    return partial(smbhb_two_masses, **p)


