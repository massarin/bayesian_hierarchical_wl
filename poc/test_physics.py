"""Rung 0/1: does the JAX lens model compute the right thing, and is it differentiable?

Every test asserts; every test leaves a plot in poc/figs/ named after itself.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.integrate import quad

from hod_mod.core.halo_profiles import nfw_sigma, nfw_delta_sigma
from hod_mod.core.lensing_profiles import nfw_params_from_mass

import jaxwl
import wl_lens_models
import mock

z_lens, z_source = mock.z_lens, mock.z_source
halos = [(1e12, 8., 3e10), (1e13, 5., 1e11), (1e14, 4., 3e11)]


@pytest.mark.parametrize('m200,c200,mstar', halos)
def test_kappa_gammat_match_numpy_reference(panel, m200, c200, mstar):
    """The JAX model must be the same physics as the trusted numpy wl_lens_models."""
    R = np.logspace(-2, 0.5, 60)
    ref = wl_lens_models.NFWPoint(z=z_lens, m200=m200, c200=c200, mstar=mstar)
    theta = R * ref.Mpc2deg  # NFWPoint takes degrees

    kappa, gammat = jaxwl.kappa_gammat(R, z_lens, jaxwl.S_cr(z_lens, z_source), m200, c200, mstar)
    dk = np.abs(np.asarray(kappa) / ref.kappa(theta, z_source) - 1.)
    dg = np.abs(np.asarray(gammat) / ref.gammat(theta, z_source) - 1.)

    panel.loglog(R, dk, label=r'$\kappa$')
    panel.loglog(R, dg, label=r'$\gamma_t$')
    panel.axhline(1e-3, color='C3', ls='--', label='tolerance')
    panel.set_xlabel('$R$ [proper Mpc]'); panel.set_ylabel('|JAX / numpy - 1|')
    panel.set_title(r'$\log M_{200}=%.0f$, worst %.1e' % (np.log10(m200), max(dk.max(), dg.max())), fontsize=9)
    panel.legend(fontsize=7)

    assert dk.max() < 1e-3
    assert dg.max() < 1e-3


def test_sigma_matches_numerical_quadrature_across_branch_cut(panel):
    """nfw_sigma switches branch at x=1 via lax.cond with a 1e-6 guard band.

    Compare the closed form against a direct line-of-sight integral of rho, straddling x=1,
    where the analytic expression suffers catastrophic cancellation.
    """
    rho_s, rs = 1e15, 1.0
    x = np.concatenate([np.linspace(0.9, 1. - 2e-6, 30), np.linspace(1. - 1e-6, 1. + 1e-6, 21),
                        np.linspace(1. + 2e-6, 1.1, 30)])

    rho = lambda r: rho_s / ((r / rs) * (1. + r / rs)**2)
    numeric = np.array([2. * quad(lambda z: rho(np.hypot(xi * rs, z)), 0., np.inf, limit=200)[0] for xi in x])
    analytic = np.asarray(nfw_sigma(jnp.array(x * rs), rho_s, rs))
    dev = np.abs(analytic / numeric - 1.)

    panel.semilogy(x, dev, 'k-', lw=1)
    panel.axvspan(1 - 1e-6, 1 + 1e-6, color='C3', alpha=.2, label='lax.cond eq1 branch')
    panel.axhline(1e-3, color='C3', ls='--', label='tolerance')
    panel.set_xlabel('$x = R/r_s$'); panel.set_ylabel(r'|$\Sigma_{\rm analytic}/\Sigma_{\rm quad} - 1$|')
    panel.set_title('closed form vs LOS quadrature\nworst %.1e' % dev.max(), fontsize=9)
    panel.legend(fontsize=7)

    assert dev.max() < 1e-3


@pytest.mark.parametrize('f', [nfw_sigma, nfw_delta_sigma], ids=['sigma', 'delta_sigma'])
def test_no_nan_gradients_across_branch_cut(panel, f):
    """vmap(lax.cond) lowers to select_n, which evaluates BOTH branches.

    _sigma_lt1 computes sqrt(1-x^2), NaN for x>1. A 0*NaN in the backward pass would
    silently poison NUTS. hod_mod's guard band avoids it; a jnp.where would not.
    """
    x = np.concatenate([np.logspace(-2, -1e-9, 40), [1.0], np.logspace(1e-9, 2, 40)])
    grad = np.array([float(jax.grad(lambda r_s: f(jnp.array([xi]), 1e15, r_s).sum())(1.0)) for xi in x])

    panel.loglog(x, np.abs(grad), '.-', ms=3)
    panel.axvline(1., color='C3', ls='--', lw=.8, label='$x=1$')
    panel.set_xlabel('$x = R/r_s$'); panel.set_ylabel(r'$|\partial f/\partial r_s|$')
    panel.set_title('%s: %d non-finite of %d' % (f.__name__, (~np.isfinite(grad)).sum(), len(x)), fontsize=9)
    panel.legend(fontsize=7)

    assert np.isfinite(grad).all()


def test_gradients_match_finite_differences(panel):
    """Finiteness is not correctness. Check d(g_t)/d(psi) against central differences."""
    R = np.logspace(-1.5, 0., 12)
    s_cr = jaxwl.S_cr(z_lens, z_source)
    g = lambda p: jaxwl.reduced_shear(R, z_lens, s_cr, 10**p[0], 10**p[1], 10**p[2]).sum()

    p0 = jnp.array([13.0, 0.7, 11.0])
    auto = np.asarray(jax.grad(g)(p0))
    eps = 1e-5
    fd = np.array([float((g(p0.at[i].add(eps)) - g(p0.at[i].add(-eps))) / (2 * eps)) for i in range(3)])
    dev = np.abs(auto / fd - 1.)

    names = [r'$\log M_{200}$', r'$\log c_{200}$', r'$\log M_*$']
    panel.bar(np.arange(3) - .17, auto, .34, label='autodiff')
    panel.bar(np.arange(3) + .17, fd, .34, label='finite diff')
    panel.set_xticks(range(3)); panel.set_xticklabels(names)
    panel.set_ylabel(r'$\partial \Sigma g_t / \partial \theta$')
    panel.set_title('autodiff vs central differences\nworst rel dev %.1e' % dev.max(), fontsize=9)
    panel.legend(fontsize=7)

    assert dev.max() < 1e-5


def test_m200_round_trip_through_rho_s_rs(panel):
    """mdef='200c' must match the repo's 200 * rho_crit, or every mass is silently offset."""
    lm = np.linspace(11.5, 15., 40)
    worst = 0.
    for c200 in [3., 5., 8.]:
        rec = []
        for m in 10**lm:
            rho_s, rs, _ = nfw_params_from_mass(m * jaxwl.h, c200, z_lens, jaxwl.cosmo, mdef='200c')
            rec.append(4. * np.pi * float(rho_s) * float(rs)**3 * (np.log(1. + c200) - c200 / (1. + c200)) / jaxwl.h)
        dev = np.abs(np.array(rec) / 10**lm - 1.)
        worst = max(worst, dev.max())
        panel.semilogy(lm, dev + 1e-18, label='c=%.0f' % c200)

    panel.set_xlabel(r'$\log M_{200}$'); panel.set_ylabel('|recovered/input - 1|')
    panel.set_title('200c mass definition round trip\nworst %.1e' % worst, fontsize=9)
    panel.legend(fontsize=7)

    assert worst < 1e-10


def test_sigma_and_sigma_cr_share_unit_conversion(panel):
    """kappa = Sigma/Sigma_cr could be right by cancellation of two unit errors. Check separately."""
    proper = lambda x: x * jaxwl.h * (1. + z_lens)**2   # comoving h-units -> proper physical
    ref = wl_lens_models.NFWPoint(z=z_lens, m200=1e13, c200=5., mstar=1e11)

    zs = np.linspace(0.35, 2.0, 40)
    scr = proper(np.array([float(jaxwl.S_cr(z_lens, z)) for z in zs])) / np.array([ref.S_cr(z) for z in zs])

    rho_s, rs, _ = nfw_params_from_mass(1e13 * jaxwl.h, 5., z_lens, jaxwl.cosmo, mdef='200c')
    R = np.logspace(-2, 0.5, 40)
    sig = proper(np.asarray(nfw_sigma(jaxwl.comoving_h(R, z_lens), rho_s, rs)))
    sig = sig / (np.atleast_1d(ref.Sigma(R * ref.Mpc2deg)) * 1e12)

    panel.semilogy(zs, np.abs(scr - 1.), label=r'$\Sigma_{cr}(z_s)$')
    panel.semilogy(zs, np.abs(sig - 1.), label=r'$\Sigma(R)$, $R$ on rescaled axis')
    panel.axhline(1e-3, color='C3', ls='--', label='tolerance')
    panel.set_xlabel('$z_{\\rm source}$'); panel.set_ylabel('|proper(JAX)/reference - 1|')
    panel.set_title(r'both convert as $\times h(1+z_l)^2$', fontsize=9)
    panel.legend(fontsize=7)

    assert np.abs(scr - 1.).max() < 1e-3
    assert np.abs(sig - 1.).max() < 1e-3


def test_population_vmap_matches_per_lens_loop(panel):
    """reduced_shear_pop vmaps over lenses; a broadcasting slip here is invisible in the posterior."""
    d = mock.make(seed=7, nlens=12, nsrc=20)
    s_cr = d['s_cr']

    batched = np.asarray(jaxwl.reduced_shear_pop(d['R'], z_lens, s_cr, 10**d['lm200'], 10**d['lc200'], 10**d['lmstar']))
    looped = np.array([np.asarray(jaxwl.reduced_shear(d['R'][i], z_lens, s_cr,
                                                      10**d['lm200'][i], 10**d['lc200'][i], 10**d['lmstar'][i]))
                       for i in range(12)])

    panel.plot(looped.ravel(), batched.ravel(), '.', ms=3)
    lim = [looped.min(), looped.max()]
    panel.plot(lim, lim, 'k--', lw=.8)
    panel.set_xlabel('$g_t$, per-lens loop'); panel.set_ylabel('$g_t$, vmap')
    panel.set_title('vmap vs loop, max |diff| %.1e' % np.abs(batched - looped).max(), fontsize=9)

    assert np.allclose(batched, looped, rtol=1e-12, atol=0.)


def test_mock_is_in_the_hierarchical_regime(panel):
    """Individual lenses must be undetected: if they were not, the hierarchy would be doing nothing."""
    snrs = {}
    for nlens in (100, 1000):
        d = mock.make(seed=42, nlens=nlens, nsrc=50)
        g = np.asarray(jaxwl.reduced_shear_pop(d['R'], z_lens, d['s_cr'],
                                               10**d['lm200'], 10**d['lc200'], 10**d['lmstar']))
        sig = 2. * mock.R_resp * g
        per_lens = np.sqrt((sig**2).sum(1)) / mock.sigma_e
        snrs[nlens] = (per_lens, np.sqrt((sig**2).sum()) / mock.sigma_e)
        panel.hist(per_lens, bins=30, histtype='step', label='N=%d, stacked S/N %.1f' % (nlens, snrs[nlens][1]))

    panel.set_xlabel('per-lens S/N'); panel.set_ylabel('lenses')
    panel.set_title('individual lenses undetected by design', fontsize=9)
    panel.legend(fontsize=7)

    assert np.median(snrs[100][0]) < 1.       # per-lens: no individual detection
    assert snrs[1000][1] > 3. * snrs[100][1] / 3.5   # stacked S/N grows ~sqrt(N)
