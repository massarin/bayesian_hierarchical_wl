# Validation ladder

Reproducing Sonnenfeld & Leauthaud (2018) by sampling `P(eta, psi_1..psi_N | d)` jointly with NUTS,
instead of marginalising each `psi_i` by Monte Carlo integration over a precomputed likelihood grid.

Independent-lens assumption retained: `P(d|eta) = prod_i P(d_i|eta)`. Joint sampling removes the
*integral*, not the *independence* — breaking that is separate work.

Run `python poc/checks.py` before trusting any posterior.

## Rung 0/1 — physics and gradients (`checks.py` -> `checks.png`)

| # | Check | Why it matters | Result |
|---|-------|----------------|--------|
| a | kappa, gamma_t vs `wl_lens_models.NFWPoint` | JAX/hod_mod model must be the same physics as the trusted numpy code | PASS, worst dev 1.9e-4 |
| b | Sigma continuous across x = R/r_s = 1 | `nfw_sigma` switches branch via `lax.cond` with a 1e-6 guard band; a discontinuity there would bias small-radius sources | PASS, max rel jump 6.5e-5 |
| c | **No NaN gradients across x = 1** | `vmap(lax.cond)` lowers to `select_n`, which evaluates *both* branches. `_sigma_lt1` computes `sqrt(1-x^2)`, NaN for x>1. A `0 * NaN` in the backward pass would silently poison NUTS. hod_mod's guard band avoids it; a `jnp.where` implementation would not | PASS, 0/162 non-finite |
| d | M200 round trip: `4 pi rho_s r_s^3 [ln(1+c) - c/(1+c)]` | confirms `mdef='200c'` matches the repo's `200 * rho_crit` definition — no mass conversion needed | PASS, worst 3.3e-16 |
| e | Sigma and Sigma_cr share the conversion `x h (1+z_l)^2` | hod_mod is comoving Mpc/h; repo is proper Mpc. Both quantities carry the same factor, so it cancels in kappa. Checked *separately* — otherwise kappa could be right by cancellation of two unit errors | PASS, worst dev 1.2e-4 |
| f | per-lens S/N << 1 | individual lenses must be undetected, so the hierarchy is doing the inference. This is SL18's regime | PASS, mean per-lens S/N 0.51 |

Residual in (a) and (e) is a constant ~2e-4 offset on *both* kappa and gamma_t: it is the Sigma_cr
comoving-distance integral (hod_mod's flat-LCDM vs the repo's `scipy.quad`), not the profile.

## Rung 2 — one lens, eta fixed (`rung2_single_lens.py`)

NUTS on `psi` reproduces the brute-force grid marginals of the same posterior. 0 divergences.
Establishes that the JAX likelihood is the object `get_wl_likelihood_grids.py` tabulates.

## Rung 3 — joint (eta, psi), N lenses (`rung3_hierarchical.py`)

Dimension 7 + 3N. Non-centred parameterisation; `--centred` reruns the same model centred, to show
the funnel rather than assert it.

Information content (from check f): stacked S/N ~5.8 at N=100, ~18 at N=1000. So N=100 constrains
`mh_mu` and `mh_beta` but leaves `mh_sig`, `c_sig` prior-dominated. All seven only at N=1000.

## Known deferrals

- `mstar_cut` selection term dropped in v1 (mock has no cut). When restored, note
  `examples/infer_nfw_shmr.py:147` compares the raw N(0,1) draw against `mstar_cut = 11.0`
  instead of `mstar_here` — the erf is ~-1 for essentially every sample.
- Sources are a fixed count per lens. Real catalogues are ragged: pad to `(N_lens, S_max)` + mask.
- Only NFW + central point mass. gNFW (the paper's M/L-gradient result) needs Sigma(R; gamma),
  which has no closed form — replace `wl_profiles/gnfw.py`'s HDF5 grids with fixed-node
  Gauss-Legendre quadrature in JAX.
- `ndinterp.py` was patched for modern scipy (`scipy.float64`, `scipy.rollaxis` removed).
