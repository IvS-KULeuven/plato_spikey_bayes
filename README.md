# Bayesian inference on the Spikey time series as seen by the PLATO mission

"Spikey" is a supermassive black hole binary candidate reported by [Hu et al. 2020](https://arxiv.org/abs/1910.05348). Following their recipe, a SMBHB is modeled as a sum of self-lensing, doppler boosting and damped random walk, the latter to simulate the red-noise-like quasar variability.

The objective is to find the parameter regime in which a spikey-like SMBHB would be recoverable from PLATO time series using bayesian inference.


To install dependencies, first install [uv](https://docs.astral.sh/uv/getting-started/installation/). After that do

```bash
uv sync
uv run jupyter notebook
```

and select one of the notebooks in the `notebook` subdirectory.

Light curves are stored in the `data` subdirectory with the format:
```
data_<source>_<instrument>_<binning>.csv
```

Bayesian inferences are stored in the `results` subdirectory with the format:
```
<source>_<instrument>_<QDM>_<binning>_<inference>_<priors>.csv
```
