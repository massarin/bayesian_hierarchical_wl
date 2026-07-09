"""Rung 3: joint NUTS over (eta, psi_1..psi_N). N=100 lenses -> 307 dimensions, no integrals."""
import sys, os, time, argparse

p = argparse.ArgumentParser()
p.add_argument('--nlens', type=int, default=100)
p.add_argument('--nsrc', type=int, default=50)
p.add_argument('--chains', type=int, default=8)
p.add_argument('--param', choices=['mixed', 'noncentred', 'centred'], default='mixed')
p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
p.add_argument('--short', action='store_true', help='quick geometry probe, not a production run')
args = p.parse_args()

# must precede any JAX backend initialisation
import numpyro
numpyro.set_platform(args.device)
if args.device == 'cpu':
    numpyro.set_host_device_count(args.chains)   # one CPU device per chain

# a single GPU holds one device: chains must be vectorised, not mapped across devices
chain_method = 'vectorized' if args.device == 'gpu' else 'parallel'

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

models = {'mixed': model.mixed, 'noncentred': model.hierarchical, 'centred': model.centred}
nwarm, ndraw = (300, 300) if args.short else (1000, 1000)
kernel = NUTS(models[args.param], target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=ndraw, num_chains=args.chains,
            chain_method=chain_method, progress_bar=False)

t0 = time.time()
mcmc.run(jax.random.PRNGKey(0), d['R'], d['s_cr'], d['et'], d['lmstar_obs'],
         extra_fields=('diverging', 'num_steps'))
dt = time.time() - t0

ndim = 7 + 3 * args.nlens
ndiv = int(mcmc.get_extra_fields()['diverging'].sum())
nsteps = np.asarray(mcmc.get_extra_fields()['num_steps'])
s = mcmc.get_samples()
names = ['mh_mu', 'mh_sig', 'mh_beta', 'ms_mu', 'ms_sig', 'c_mu', 'c_sig']

# hyper-parameters only: the 3N latents are nuisance
from numpyro.diagnostics import effective_sample_size, gelman_rubin
chained = mcmc.get_samples(group_by_chain=True)

# saturating max tree depth means the step size is throttled by a badly scaled direction:
# a geometry problem, invisible in the divergence count
saturating = 100. * (nsteps >= 1023).mean()
min_ess = min(float(effective_sample_size(np.asarray(chained[k]))) for k in names)
print('%s: %d lenses, %d dims, %.1f s, %d divergences' % (args.param, args.nlens, ndim, dt, ndiv))
print('  median %d leapfrogs/draw, %.0f%% saturating 2^10, min ESS %.0f (%.1f ESS/s)' % (
    np.median(nsteps), saturating, min_ess, min_ess / dt))
print('\n%-9s %8s %8s %9s %8s %7s %7s' % ('', 'truth', 'mean', 'std', 'z', 'n_eff', 'r_hat'))
for k in names:
    x = np.asarray(s[k])
    z = (x.mean() - truth[k]) / x.std()
    print('%-9s %8.3f %8.3f %9.3f %8.1f %7.0f %7.3f' % (
        k, truth[k], x.mean(), x.std(), z,
        effective_sample_size(np.asarray(chained[k])), gelman_rubin(np.asarray(chained[k]))))
worst_z = max(abs((np.asarray(s[k]).mean() - truth[k]) / np.asarray(s[k]).std()) for k in names)
print('\nworst |z| vs truth: %.2f' % worst_z)

fig, axes = plt.subplots(1, 7, figsize=(17, 2.8), constrained_layout=True)
for ax, k in zip(axes, names):
    x = np.asarray(s[k])
    ax.hist(x, bins=40, density=True, histtype='stepfilled', color='C0', alpha=0.5)
    ax.axvline(truth[k], color='k', ls='--', lw=1.5)
    ax.set_title('%s\n%.3f $\\pm$ %.3f' % (k, x.mean(), x.std()), fontsize=9)
    ax.set_yticks([])
fig.suptitle('%d lenses, %d dimensions, %d divergences (dashed = truth)' % (args.nlens, ndim, ndiv), fontsize=10)

tag = '%s_n%d' % (args.param, args.nlens)
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
