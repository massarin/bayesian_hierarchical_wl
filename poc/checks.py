"""Rung 0/1 diagnostics: everything the sampler silently assumes. One panel per assumption.

Run before trusting any posterior. Writes checks.png and prints PASS/FAIL per check.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hod_mod.core.halo_profiles import nfw_sigma, nfw_delta_sigma
from hod_mod.core.lensing_profiles import nfw_params_from_mass
import jaxwl
import wl_lens_models
import mock

results = []
report = lambda name, ok, detail: results.append((name, bool(ok), detail))

fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), constrained_layout=True)
z_lens, z_source = mock.z_lens, mock.z_source
s_cr = jaxwl.S_cr(z_lens, z_source)


# (a) profiles agree with the trusted numpy reference
ax = axes[0, 0]
R = np.logspace(-2, 0.5, 60)
worst = 0.
for (m200, c200, mstar), col in zip([(1e12, 8., 3e10), (1e13, 5., 1e11), (1e14, 4., 3e11)],
                                    plt.cm.viridis(np.linspace(.1, .8, 3))):
    ref = wl_lens_models.NFWPoint(z=z_lens, m200=m200, c200=c200, mstar=mstar)
    theta = R * ref.Mpc2deg
    kappa, gammat = jaxwl.kappa_gammat(R, z_lens, s_cr, m200, c200, mstar)
    rk = np.asarray(kappa) / ref.kappa(theta, z_source)
    rg = np.asarray(gammat) / ref.gammat(theta, z_source)
    worst = max(worst, np.abs(np.r_[rk, rg] - 1.).max())
    ax.semilogx(R, rk, '-', color=col, label=r'$\kappa$, $\log M=%.0f$' % np.log10(m200))
    ax.semilogx(R, rg, ':', color=col)
ax.axhline(1., color='k', lw=.5); ax.set_ylim(.99, 1.01)
ax.set_xlabel('$R$ [proper Mpc]'); ax.set_ylabel('JAX / numpy reference')
ax.set_title('(a) profiles vs wl_lens_models\nworst dev %.2e' % worst, fontsize=9)
ax.legend(fontsize=6)
report('profiles match NFWPoint (<1%)', worst < 1e-2, 'worst |ratio-1| = %.2e' % worst)


# (b) Sigma is continuous across the x=1 branch cut (lt1 / eq1 / gt1 in lax.cond)
ax = axes[0, 1]
rho_s, rs = 1e15, 1.0
x = np.linspace(1 - 3e-6, 1 + 3e-6, 401)
sig = np.asarray(nfw_sigma(jnp.array(x * rs), rho_s, rs))
jump = np.abs(np.diff(sig)).max() / np.abs(sig).mean()
ax.plot(x, sig / sig.mean(), 'k-', lw=1)
ax.axvspan(1 - 1e-6, 1 + 1e-6, color='C3', alpha=.2, label='lax.cond eq1 branch')
ax.set_xlabel('$x = R/r_s$'); ax.set_ylabel(r'$\Sigma / \langle\Sigma\rangle$')
ax.set_title('(b) branch continuity at $x=1$\nmax rel. jump %.1e' % jump, fontsize=9)
ax.legend(fontsize=6)
report('Sigma continuous across x=1', jump < 1e-4, 'max rel jump = %.2e' % jump)


# (c) gradients finite for x<1, x=1, x>1. vmap(lax.cond) -> select_n evaluates both
#     branches, and sqrt(1-x^2) is NaN for x>1: a 0*NaN here would poison NUTS.
ax = axes[0, 2]
xg = np.concatenate([np.logspace(-2, -0.01, 40), [1.0], np.logspace(0.01, 2, 40)])
dS = np.array([float(jax.grad(lambda r: nfw_sigma(jnp.array([xi * 1.0]), 1e15, r).sum())(1.0)) for xi in xg])
dDS = np.array([float(jax.grad(lambda r: nfw_delta_sigma(jnp.array([xi * 1.0]), 1e15, r).sum())(1.0)) for xi in xg])
nbad = int((~np.isfinite(dS)).sum() + (~np.isfinite(dDS)).sum())
ax.loglog(xg, np.abs(dS), '.-', ms=3, label=r'$|\partial\Sigma/\partial r_s|$')
ax.loglog(xg, np.abs(dDS), '.-', ms=3, label=r'$|\partial\Delta\Sigma/\partial r_s|$')
ax.axvline(1., color='C3', ls='--', lw=.8)
ax.set_xlabel('$x = R/r_s$'); ax.set_title('(c) gradient finiteness across $x=1$\n%d NaN/Inf' % nbad, fontsize=9)
ax.legend(fontsize=6)
report('no NaN gradients across x=1', nbad == 0, '%d non-finite of %d' % (nbad, 2 * len(xg)))


# (d) mass definition round trip: M200 = 4 pi rho_s rs^3 [ln(1+c) - c/(1+c)]
ax = axes[1, 0]
lm = np.linspace(11.5, 15., 40)
err = []
for c200 in [3., 5., 8.]:
    rec = []
    for m in 10**lm:
        rho_s, rs, _ = nfw_params_from_mass(m * jaxwl.h, c200, z_lens, jaxwl.cosmo, mdef='200c')
        mrec = 4. * np.pi * float(rho_s) * float(rs)**3 * (np.log(1. + c200) - c200 / (1. + c200))
        rec.append(mrec / jaxwl.h)
    rec = np.array(rec)
    err.append(np.abs(rec / 10**lm - 1.).max())
    ax.semilogy(lm, np.abs(rec / 10**lm - 1.) + 1e-17, label='c=%.0f' % c200)
ax.set_xlabel(r'$\log M_{200}$'); ax.set_ylabel('|recovered/input - 1|')
ax.set_title('(d) 200c mass definition round trip\nworst %.1e' % max(err), fontsize=9)
ax.legend(fontsize=6)
report('M200 round trip through rho_s, r_s', max(err) < 1e-8, 'worst = %.2e' % max(err))


# (e) Sigma_crit and Sigma agree with the repo cosmology, and do so under the *same*
#     unit conversion -- otherwise kappa = Sigma/Sigma_cr could be right by cancellation.
ax = axes[1, 1]
proper = lambda x: x * jaxwl.h * (1. + z_lens)**2   # comoving h-units -> proper physical

ref = wl_lens_models.NFWPoint(z=z_lens, m200=1e13, c200=5., mstar=1e11)
zs = np.linspace(0.35, 2.0, 40)
scr_ratio = proper(np.array([float(jaxwl.S_cr(z_lens, z)) for z in zs])) / np.array([ref.S_cr(z) for z in zs])

rho_s, rs, _ = nfw_params_from_mass(1e13 * jaxwl.h, 5., z_lens, jaxwl.cosmo, mdef='200c')
Rs = np.logspace(-2, 0.5, 40)
sig_jax = proper(np.asarray(nfw_sigma(jaxwl.comoving_h(Rs, z_lens), rho_s, rs)))
sig_ratio = sig_jax / (np.atleast_1d(ref.Sigma(Rs * ref.Mpc2deg)) * 1e12)

ax.plot(zs, scr_ratio, 'C0-', label=r'$\Sigma_{cr}(z_s)$')
ax.plot(np.linspace(0.35, 2.0, 40), sig_ratio, 'C1-', label=r'$\Sigma(R)$ (x-axis: $R$ rescaled)')
ax.axhline(1., color='C3', ls='--', lw=.8); ax.set_ylim(.995, 1.005)
ax.set_xlabel('$z_{\\rm source}$'); ax.set_ylabel('proper(JAX) / reference')
worst_u = max(np.abs(scr_ratio - 1.).max(), np.abs(sig_ratio - 1.).max())
ax.set_title(r'(e) $\Sigma$ and $\Sigma_{cr}$ share the conversion' + '\n$\\times h(1+z_l)^2$, worst dev %.1e' % worst_u, fontsize=9)
ax.legend(fontsize=6)
report('Sigma & Sigma_cr units consistent (<0.5%)', worst_u < 5e-3, 'worst |ratio-1| = %.2e' % worst_u)


# (f) information content of the mock: is there anything to infer?
ax = axes[1, 2]
for nlens, nsrc, col in [(100, 50, 'C0'), (1000, 50, 'C1')]:
    d = mock.make(seed=42, nlens=nlens, nsrc=nsrc)
    g = np.asarray(jaxwl.reduced_shear_pop(d['R'], z_lens, d['s_cr'], 10**d['lm200'], 10**d['lc200'], 10**d['lmstar']))
    sig = 2. * mock.R_resp * g
    snr_lens = np.sqrt((sig**2).sum(1)) / mock.sigma_e
    stacked = np.sqrt((sig**2).sum()) / mock.sigma_e
    ax.hist(snr_lens, bins=30, histtype='step', color=col, label='N=%d, stacked S/N %.1f' % (nlens, stacked))
ax.set_xlabel('per-lens S/N'); ax.set_ylabel('lenses')
ax.set_title('(f) mock information content\nindividual lenses undetected by design', fontsize=9)
ax.legend(fontsize=6)
report('per-lens S/N << 1 (hierarchy does the work)', snr_lens.mean() < 1., 'mean per-lens S/N = %.2f' % snr_lens.mean())

plt.savefig(os.path.join(os.path.dirname(__file__), 'checks.png'), dpi=110)

print()
npass = sum(ok for _, ok, _ in results)
for name, ok, detail in results:
    print('  [%s] %-42s %s' % ('PASS' if ok else 'FAIL', name, detail))
print('\n%d/%d checks passed -> checks.png' % (npass, len(results)))
sys.exit(0 if npass == len(results) else 1)
