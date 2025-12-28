import numpy as np
from jax import numpy as jnp
from corner import corner


def corner_plot(samples, names=None, true_params=None):
    if names is None:
        names = [k for k in samples.keys() if 'pred' not in k and 'eta' not in k]
    fig_corner = corner(
        data=np.stack([samples[k] for k in names]).T, bins=30, labels=names, show_titles=True, 
    );
    ndim = len(names)
    axes = np.array(fig_corner.axes).reshape((ndim, ndim))
    
    if true_params is not None:
        for i in range(ndim):
            ax = axes[i, i]
            if names[i] in true_params:
                ax.axvline(true_params[names[i]], color="g")
        
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                if names[xi] in true_params:
                    ax.axvline(true_params[names[xi]], color="g")
                if names[yi] in true_params:
                    ax.axhline(true_params[names[yi]], color="g")  

def predictive_posterior(ax, x, samples):
    if 'pred_gp_mean' in samples:
        gp_med_mean = jnp.median(samples["pred_gp_mean"], axis=0)
        gp_med_std = jnp.median(samples["pred_gp_std"], axis=0)
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

