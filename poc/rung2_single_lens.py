"""Rung 2: one lens, eta fixed. NUTS on psi vs brute-force evaluation of the same likelihood on a grid.

Confirms the JAX likelihood is the same object the old pipeline tabulates in get_wl_likelihood_grids.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

import mock
from mock import truth, mstar_piv, z_lens, mstar_err, sigma_e
from model import et_model

d = mock.make(seed=1, nlens=1, nsrc=2000)   # many sources: one lens, well constrained
s_cr, R, et, lmstar_obs = d['s_cr'], d['R'], d['et'], d['lmstar_obs']


def logpost(lmstar, lm200, lc200):
    """psi-posterior at fixed eta: P(d|psi) P(psi|eta)."""
    ll = -0.5 * (((et - et_model(R, s_cr, lmstar, lm200, lc200)) / sigma_e)**2).sum()
    ll += -0.5 * ((lmstar_obs - lmstar) / mstar_err)**2
    ll += -0.5 * ((lmstar - truth['ms_mu']) / truth['ms_sig'])**2
    ll += -0.5 * ((lm200 - truth['mh_mu'] - truth['mh_beta'] * (lmstar - mstar_piv)) / truth['mh_sig'])**2
    ll += -0.5 * ((lc200 - truth['c_mu']) / truth['c_sig'])**2
    return ll.sum()


def psi_model(R, s_cr, et, lmstar_obs):
    lmstar = numpyro.sample('lmstar', dist.Normal(truth['ms_mu'], truth['ms_sig']), sample_shape=(1,))
    lm200 = numpyro.sample('lm200', dist.Normal(truth['mh_mu'] + truth['mh_beta'] * (lmstar - mstar_piv), truth['mh_sig']))
    lc200 = numpyro.sample('lc200', dist.Normal(truth['c_mu'], truth['c_sig']), sample_shape=(1,))
    numpyro.sample('mstar_obs', dist.Normal(lmstar, mstar_err), obs=lmstar_obs)
    numpyro.sample('shapes', dist.Normal(et_model(R, s_cr, lmstar, lm200, lc200), sigma_e), obs=et)


mcmc = MCMC(NUTS(psi_model), num_warmup=500, num_samples=2000, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), R, s_cr, et, lmstar_obs, extra_fields=('diverging',))
s = mcmc.get_samples()
print('divergences:', int(mcmc.get_extra_fields()['diverging'].sum()))

# brute force: the marginals of the same posterior, on a grid
grids = {'lmstar': np.linspace(10.5, 12.0, 61), 'lm200': np.linspace(12.0, 14.5, 81), 'lc200': np.linspace(0.3, 1.1, 61)}
mesh = np.meshgrid(*grids.values(), indexing='ij')
lp = np.array(jax.vmap(lambda a, b, c: logpost(jnp.array([a]), jnp.array([b]), jnp.array([c])))(
    *[jnp.array(m.ravel()) for m in mesh])).reshape(mesh[0].shape)
post = np.exp(lp - lp.max())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
for ax, (k, g), axis in zip(axes, grids.items(), [(1, 2), (0, 2), (0, 1)]):
    marg = post.sum(axis=axis)
    ax.plot(g, marg / np.trapezoid(marg, g), 'k-', lw=3, alpha=0.4, label='grid (brute force)')
    ax.hist(np.asarray(s[k]).ravel(), bins=40, density=True, histtype='step', color='C3', label='NUTS')
    ax.axvline(d[k][0], color='C0', ls=':', label='truth')
    ax.set_xlabel(k); ax.legend(fontsize=7)
axes[0].set_ylabel('posterior density')
axes[1].set_title('one lens, 2000 sources, $\\eta$ fixed')
plt.savefig(os.path.join(os.path.dirname(__file__), 'rung2_single_lens.png'), dpi=110)
print('wrote rung2_single_lens.png')
