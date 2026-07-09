"""Mock lens population and weak-lensing observables, following Sonnenfeld & Leauthaud (2018) sec 3.2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jaxwl

# hyper-parameters eta, at the values of the SL18 control-sample inference (their table 1)
truth = dict(mh_mu=13.0, mh_sig=0.30, mh_beta=1.5, ms_mu=11.3, ms_sig=0.25, c_mu=0.7, c_sig=0.10)

mstar_piv = 11.3
z_lens, z_source = 0.3, 0.9

mstar_err = 0.15     # dex, on the observed stellar mass
sigma_e = 0.3        # per-source shape noise
R_resp = 0.867       # shear responsivity, 1 - e_rms^2
rmin, rmax = 0.05, 0.5   # proper Mpc, radial range of the source sample


def draw_lenses(rng, nlens, eta=truth):
    """psi = (logM*, logM200, logc200) drawn from P(psi|eta), plus a noisy logM* measurement."""
    lmstar = rng.normal(eta['ms_mu'], eta['ms_sig'], nlens)
    lm200 = rng.normal(eta['mh_mu'] + eta['mh_beta'] * (lmstar - mstar_piv), eta['mh_sig'])
    lc200 = rng.normal(eta['c_mu'], eta['c_sig'], nlens)

    lmstar_obs = lmstar + rng.normal(0., mstar_err, nlens)
    return lmstar, lm200, lc200, lmstar_obs


def draw_sources(rng, nlens, nsrc):
    """Source positions, uniform in area over the annulus [rmin, rmax]."""
    u = rng.uniform(size=(nlens, nsrc))
    return (rmin**2 + u * (rmax**2 - rmin**2))**0.5


def draw_ellipticities(rng, R, lmstar, lm200, lc200):
    """e_t = 2 R_resp * g_t + noise  (SL18 eq 20, with m = c_t = 0)."""
    s_cr = jaxwl.S_cr(z_lens, z_source)
    g = jaxwl.reduced_shear_pop(R, z_lens, s_cr, 10**lm200, 10**lc200, 10**lmstar)
    return 2. * R_resp * np.asarray(g) + rng.normal(0., sigma_e, R.shape)


def make(seed, nlens, nsrc):
    rng = np.random.default_rng(seed)
    lmstar, lm200, lc200, lmstar_obs = draw_lenses(rng, nlens)
    R = draw_sources(rng, nlens, nsrc)
    et = draw_ellipticities(rng, R, lmstar, lm200, lc200)
    return dict(R=R, et=et, lmstar_obs=lmstar_obs, s_cr=float(jaxwl.S_cr(z_lens, z_source)),
                lmstar=lmstar, lm200=lm200, lc200=lc200)
