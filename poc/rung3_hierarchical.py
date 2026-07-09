"""Rung 3: joint NUTS over (eta, psi_1..psi_N). N=100 lenses -> 307 dimensions, no integrals."""
import sys, os, time, argparse

p = argparse.ArgumentParser()
p.add_argument('--nlens', type=int, default=100)
p.add_argument('--nsrc', type=int, default=50)
p.add_argument('--chains', type=int, default=8)
p.add_argument('--centred', action='store_true', help='demonstrate the funnel')
args = p.parse_args()

# must precede any JAX backend initialisation
import numpyro
numpyro.set_host_device_count(args.chains)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

import mock
from mock import truth
import model

print('jax devices: %d' % jax.local_device_count())
d = mock.make(seed=42, nlens=args.nlens, nsrc=args.nsrc)

kernel = NUTS(model.centred if args.centred else model.hierarchical, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=args.chains,
            chain_method='parallel', progress_bar=False)

t0 = time.time()
mcmc.run(jax.random.PRNGKey(0), d['R'], d['s_cr'], d['et'], d['lmstar_obs'], extra_fields=('diverging',))
dt = time.time() - t0

ndim = 7 + 3 * args.nlens
ndiv = int(mcmc.get_extra_fields()['diverging'].sum())
print('%s: %d lenses, %d dims, %.1f s, %d divergences' % (
    'centred' if args.centred else 'non-centred', args.nlens, ndim, dt, ndiv))
mcmc.print_summary(exclude_deterministic=True)

s = mcmc.get_samples()
names = ['mh_mu', 'mh_sig', 'mh_beta', 'ms_mu', 'ms_sig', 'c_mu', 'c_sig']

fig, axes = plt.subplots(1, 7, figsize=(17, 2.8), constrained_layout=True)
for ax, k in zip(axes, names):
    x = np.asarray(s[k])
    ax.hist(x, bins=40, density=True, histtype='stepfilled', color='C0', alpha=0.5)
    ax.axvline(truth[k], color='k', ls='--', lw=1.5)
    ax.set_title('%s\n%.3f $\\pm$ %.3f' % (k, x.mean(), x.std()), fontsize=9)
    ax.set_yticks([])
fig.suptitle('%d lenses, %d dimensions, %d divergences (dashed = truth)' % (args.nlens, ndim, ndiv), fontsize=10)

tag = 'centred' if args.centred else 'noncentred'
plt.savefig(os.path.join(os.path.dirname(__file__), 'rung3_eta_%s.png' % tag), dpi=110)

# per-lens shrinkage: the marginalised scheme throws these away
fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
lm200 = np.asarray(s['lm200'])
ax.errorbar(d['lm200'], lm200.mean(0), yerr=lm200.std(0), fmt='.', ms=3, lw=0.5, alpha=0.6)
lo, hi = d['lm200'].min() - 0.2, d['lm200'].max() + 0.2
ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
ax.set_xlabel('true $\\log M_{200}$'); ax.set_ylabel('inferred $\\log M_{200}$')
ax.set_title('per-lens posteriors (free byproduct)', fontsize=10)
plt.savefig(os.path.join(os.path.dirname(__file__), 'rung3_perlens_%s.png' % tag), dpi=110)
print('wrote rung3_*_%s.png' % tag)
