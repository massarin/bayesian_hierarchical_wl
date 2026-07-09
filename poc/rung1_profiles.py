"""Rung 1: JAX/hod_mod NFW+point-mass vs the reference numpy wl_lens_models.NFWPoint."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import jaxwl
import wl_lens_models

z_lens, z_source = 0.3, 0.9
R = np.logspace(-2, 0.5, 60)  # proper Mpc

cases = [(1e13, 5., 1e11), (1e14, 4., 3e11), (1e12, 8., 3e10)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(cases)))

for (m200, c200, mstar), col in zip(cases, colors):
    ref = wl_lens_models.NFWPoint(z=z_lens, m200=m200, c200=c200, mstar=mstar)
    theta = R * ref.Mpc2deg  # NFWPoint takes angles in degrees

    ref_kappa = ref.kappa(theta, z_source)
    ref_gammat = ref.gammat(theta, z_source)

    s_cr = jaxwl.S_cr(z_lens, z_source)
    kappa, gammat = jaxwl.kappa_gammat(R, z_lens, s_cr, m200, c200, mstar)

    label = r'$M_{200}=10^{%.0f}$' % np.log10(m200)
    axes[0].loglog(R, ref_kappa, '-', color=col, lw=3, alpha=0.35)
    axes[0].loglog(R, kappa, '--', color=col, label=label)
    axes[1].loglog(R, ref_gammat, '-', color=col, lw=3, alpha=0.35)
    axes[1].loglog(R, gammat, '--', color=col, label=label)
    axes[2].semilogx(R, kappa / ref_kappa, '-', color=col, label=r'$\kappa$')
    axes[2].semilogx(R, gammat / ref_gammat, ':', color=col, label=r'$\gamma_t$')

    print('logM200=%4.1f  kappa ratio %.4f-%.4f   gammat ratio %.4f-%.4f' % (
        np.log10(m200), (kappa / ref_kappa).min(), (kappa / ref_kappa).max(),
        (gammat / ref_gammat).min(), (gammat / ref_gammat).max()))

axes[0].set_ylabel(r'$\kappa$'); axes[1].set_ylabel(r'$\gamma_t$')
axes[2].set_ylabel('JAX / reference'); axes[2].axhline(1., color='k', lw=0.5)
axes[2].set_ylim(0.9, 1.1)
for ax in axes:
    ax.set_xlabel(r'$R$ [proper Mpc]'); ax.legend(fontsize=7)
axes[0].set_title('thick = numpy reference, dashed = JAX/hod_mod')

plt.savefig(os.path.join(os.path.dirname(__file__), 'rung1_profiles.png'), dpi=110)
print('wrote rung1_profiles.png')
