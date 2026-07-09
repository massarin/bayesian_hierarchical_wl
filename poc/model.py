"""Joint posterior P(eta, psi_1..psi_N | d): no marginalisation integral, no likelihood grid."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import jaxwl
from mock import mstar_piv, z_lens, mstar_err, sigma_e, R_resp


def et_model(R, s_cr, lmstar, lm200, lc200):
    g = jaxwl.reduced_shear_pop(R, z_lens, s_cr, 10**lm200, 10**lc200, 10**lmstar)
    return 2. * R_resp * g


def hierarchical(R, s_cr, et, lmstar_obs):
    nlens = len(lmstar_obs)

    mh_mu = numpyro.sample('mh_mu', dist.Uniform(11., 15.))
    mh_sig = numpyro.sample('mh_sig', dist.Uniform(0., 1.))
    mh_beta = numpyro.sample('mh_beta', dist.Uniform(0., 5.))
    ms_mu = numpyro.sample('ms_mu', dist.Uniform(10., 12.))
    ms_sig = numpyro.sample('ms_sig', dist.Uniform(0., 1.))
    c_mu = numpyro.sample('c_mu', dist.Uniform(0., 2.))
    c_sig = numpyro.sample('c_sig', dist.Uniform(0., 1.))

    # non-centred: sigma -> 0 would otherwise pinch psi against eta (Neal's funnel)
    with numpyro.plate('lens', nlens):
        z_ms = numpyro.sample('z_ms', dist.Normal(0., 1.))
        z_mh = numpyro.sample('z_mh', dist.Normal(0., 1.))
        z_c = numpyro.sample('z_c', dist.Normal(0., 1.))

    lmstar = numpyro.deterministic('lmstar', ms_mu + ms_sig * z_ms)
    lm200 = numpyro.deterministic('lm200', mh_mu + mh_beta * (lmstar - mstar_piv) + mh_sig * z_mh)
    lc200 = numpyro.deterministic('lc200', c_mu + c_sig * z_c)

    numpyro.sample('mstar_obs', dist.Normal(lmstar, mstar_err), obs=lmstar_obs)
    numpyro.sample('shapes', dist.Normal(et_model(R, s_cr, lmstar, lm200, lc200), sigma_e), obs=et)


def mixed(R, s_cr, et, lmstar_obs):
    """lmstar centred, halo parameters non-centred.

    Non-centring only pays when the data are uninformative about the latent. logM* is pinned by
    lmstar_obs (0.15 dex vs 0.25 dex population scatter), so it wants the centred form; logM200
    and logc200 are constrained only by weak lensing at per-lens S/N ~ 0.5, so they want
    non-centred.
    """
    nlens = len(lmstar_obs)

    mh_mu = numpyro.sample('mh_mu', dist.Uniform(11., 15.))
    mh_sig = numpyro.sample('mh_sig', dist.Uniform(0., 1.))
    mh_beta = numpyro.sample('mh_beta', dist.Uniform(0., 5.))
    ms_mu = numpyro.sample('ms_mu', dist.Uniform(10., 12.))
    ms_sig = numpyro.sample('ms_sig', dist.Uniform(0., 1.))
    c_mu = numpyro.sample('c_mu', dist.Uniform(0., 2.))
    c_sig = numpyro.sample('c_sig', dist.Uniform(0., 1.))

    with numpyro.plate('lens', nlens):
        lmstar = numpyro.sample('lmstar', dist.Normal(ms_mu, ms_sig))
        z_mh = numpyro.sample('z_mh', dist.Normal(0., 1.))
        z_c = numpyro.sample('z_c', dist.Normal(0., 1.))

    lm200 = numpyro.deterministic('lm200', mh_mu + mh_beta * (lmstar - mstar_piv) + mh_sig * z_mh)
    lc200 = numpyro.deterministic('lc200', c_mu + c_sig * z_c)

    numpyro.sample('mstar_obs', dist.Normal(lmstar, mstar_err), obs=lmstar_obs)
    numpyro.sample('shapes', dist.Normal(et_model(R, s_cr, lmstar, lm200, lc200), sigma_e), obs=et)


def centred(R, s_cr, et, lmstar_obs):
    """Same model, fully centred. Kept to demonstrate the funnel."""
    nlens = len(lmstar_obs)

    mh_mu = numpyro.sample('mh_mu', dist.Uniform(11., 15.))
    mh_sig = numpyro.sample('mh_sig', dist.Uniform(0., 1.))
    mh_beta = numpyro.sample('mh_beta', dist.Uniform(0., 5.))
    ms_mu = numpyro.sample('ms_mu', dist.Uniform(10., 12.))
    ms_sig = numpyro.sample('ms_sig', dist.Uniform(0., 1.))
    c_mu = numpyro.sample('c_mu', dist.Uniform(0., 2.))
    c_sig = numpyro.sample('c_sig', dist.Uniform(0., 1.))

    with numpyro.plate('lens', nlens):
        lmstar = numpyro.sample('lmstar', dist.Normal(ms_mu, ms_sig))
        lm200 = numpyro.sample('lm200', dist.Normal(mh_mu + mh_beta * (lmstar - mstar_piv), mh_sig))
        lc200 = numpyro.sample('lc200', dist.Normal(c_mu, c_sig))

    numpyro.sample('mstar_obs', dist.Normal(lmstar, mstar_err), obs=lmstar_obs)
    numpyro.sample('shapes', dist.Normal(et_model(R, s_cr, lmstar, lm200, lc200), sigma_e), obs=et)
