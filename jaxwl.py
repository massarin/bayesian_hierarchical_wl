"""JAX weak-lensing model: NFW halo + central point mass. Sonnenfeld & Leauthaud (2018)."""
import jax
import jax.numpy as jnp
from hod_mod.core.halo_profiles import nfw_sigma, nfw_delta_sigma
from hod_mod.core.lensing_profiles import nfw_params_from_mass
from hod_mod.observables.lensing import sigma_crit

h = 0.7
cosmo = {'h': h, 'Omega_m': 0.3}

# hod_mod works in comoving Mpc/h and M_Sun/h; the lens catalog is in proper Mpc and M_Sun.
# Sigma and Sigma_cr both carry a factor h*(1+z_lens)**2 relative to proper units, so it
# cancels in kappa and gamma_t and only the radii need converting.
comoving_h = lambda R_proper, z: R_proper * (1. + z) * h


def S_cr(z_lens, z_source):
    return sigma_crit(z_lens, z_source, cosmo, comoving=True)


def kappa_gammat(R, z_lens, s_cr, m200, c200, mstar):
    """Convergence and tangential shear at proper radii R [Mpc] for one NFW halo + point mass.

    R is 1d; m200, c200, mstar are scalars (hod_mod's nfw_sigma vmaps over R internally).
    """
    x = comoving_h(R, z_lens)
    rho_s, rs, _ = nfw_params_from_mass(m200 * h, c200, z_lens, cosmo, mdef='200c')

    Sigma = nfw_sigma(x, rho_s, rs)
    DeltaSigma = nfw_delta_sigma(x, rho_s, rs) + mstar * h / (jnp.pi * x**2)

    return Sigma / s_cr, DeltaSigma / s_cr


def reduced_shear(R, z_lens, s_cr, m200, c200, mstar):
    kappa, gammat = kappa_gammat(R, z_lens, s_cr, m200, c200, mstar)
    return gammat / (1. - kappa)


# over a population: R is (nlens, nsrc), halo parameters are (nlens,)
reduced_shear_pop = jax.vmap(reduced_shear, in_axes=(0, None, None, 0, 0, 0))
