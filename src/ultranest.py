# Built-in
import sys
import copy

# Dependencies
import scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import jax
from jax import random
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
# Select number of CPUs
numpyro.enable_x64()
numpyro.set_host_device_count(6)

# Internal methods
sys.path.append('../src')
from smbhb_jax import smbhb_jax
import smbhb as smbhb
import bayes as bs
import utils as ut
import plots as pt
pt.setup()

# Global paths
path = Path('../')
ddir = path / 'data'
rdir = path / 'results'

# Load Hu+2020 binned light curve of Spikey 
df = pd.read_csv(ddir / 'data_spikey_DM_kepler_hu2020.csv')
time = df.time.to_numpy()

# Generate relativistic Spikey model
params = smbhb.model_params()
params.sigma = 0 
model = smbhb.model(params)
dm = model.light_curve(time, df=True)

# Plot light curve
pt.plot_lc(df, dm, cm=c);

filename = 'spikey_kepler_H20_DM_ultranest'

priors = bs.model_priors()
priors.z     = params.z
priors.t0    = [params.t0*0.95, params.t0*1.05]
priors.P     = [params.P*0.95, params.P*1.05]
priors.i     = [0, 90]
priors.e     = [params.e*0.9, params.e*1.1]
priors.w     = [0, 180]
priors.logM1 = [5, 11]
priors.logM2 = [5, 11]
priors.L     = [0, 1]
priors.alpha = [params.alpha*0.9, params.alpha*1.1]
priors.vz    = params.vz

# Run UltraNest analysis
result, sampler = bs.run_ultranest(df, params, priors, path=rdir/filename)
