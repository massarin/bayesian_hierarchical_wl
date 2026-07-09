# Joint sampling proof of concept

Reproduce Sonnenfeld & Leauthaud (2018) by sampling `P(eta, psi_1..psi_N | d)` jointly with NUTS,
instead of marginalising each `psi_i = (logM*, logM200, logc200)` by Monte Carlo integration over a
precomputed weak-lensing likelihood grid.

The independent-lens assumption is retained: `P(d|eta) = prod_i P(d_i|eta)`. Joint sampling removes
the *integral*, not the *independence*. Breaking independence (2-halo term, correlated shear) is
separate work, which `hod_mod` is here to enable.

## Layout

| file | what |
|---|---|
| `../jaxwl.py` | the physics: NFW + central point mass -> kappa, gamma_t. 30 lines |
| `mock.py` | population + sources drawn from known `eta` (SL18 table 1, control sample) |
| `model.py` | three parameterisations of the same numpyro model |
| `test_physics.py` | rung 0/1 — asserts, and plots each check into `figs/` |
| `rung2_single_lens.py` | rung 2 — NUTS on `psi` vs brute-force grid, `eta` fixed |
| `rung3_hierarchical.py` | rung 3 — joint NUTS, 7 + 3N dims |

`pytest poc/ -q` runs the physics tests. Each writes `poc/figs/<test_name>.png`, so a failure
comes with a picture.

## Results

**Rung 1.** kappa and gamma_t match the numpy `wl_lens_models.NFWPoint` to <2e-4. Gradients are
finite everywhere including x = R/r_s = 1, and match central differences.

**Rung 2.** NUTS on `psi` reproduces the brute-force grid marginals exactly. 0 divergences.
The JAX likelihood is the object `examples/get_wl_likelihood_grids.py` tabulates.

**Rung 3.** All seven hyper-parameters recovered at N=1000 (3007 dims, 0 divergences, r_hat = 1.000,
1000 warmup + 1000 draws x 8 chains):

| | truth | inferred | z |
|---|---|---|---|
| `mh_mu` | 13.000 | 12.827 +/- 0.117 | -1.5 |
| `mh_sig` | 0.300 | 0.289 +/- 0.123 | -0.1 |
| `mh_beta` | 1.500 | 1.536 +/- 0.261 | +0.1 |
| `ms_mu` | 11.300 | 11.293 +/- 0.009 | -0.8 |
| `ms_sig` | 0.250 | 0.247 +/- 0.008 | -0.4 |
| `c_mu` | 0.700 | 0.710 +/- 0.142 | +0.1 |
| `c_sig` | 0.100 | 0.206 +/- 0.148 | +0.7 |

At N=100, `mh_sig = 0.39 +/- 0.25` is essentially the Uniform(0,1) prior with a bump, as the stacked
S/N predicts. It only becomes informative at N=1000. `c_sig` stays prior-dominated even there:
concentration is the direction weak lensing constrains worst.

`rung3_perlens_mixed_n1000.png` shows the per-lens logM200 posteriors, a free byproduct of joint
sampling that the marginalised scheme discards. The slope is visibly shallower than unity: shrinkage
towards the population mean.

## Parameterisation: mixed, not non-centred

Non-centring only pays when the data are *un*informative about the latent. `logM*` is pinned by
`lmstar_obs` (0.15 dex, against 0.25 dex population scatter), so it wants the **centred** form.
`logM200` and `logc200` are constrained only by weak lensing at per-lens S/N ~ 0.5, so they want
**non-centred**.

At N=1000, 3007 dims, same seed and same sampler settings:

| | time | divergences | saturating 2^10 | min ESS | worst r_hat |
|---|---|---|---|---|---|
| **mixed** | **493 s** | **0** | **0%** | **854** | **1.002** |
| fully non-centred | 888 s | 97 | 42% | 2 | 6.8 |
| fully centred | 1011 s | 0 | 25% | 2 | 11.9 |

The failure is silent. The fully centred run reports **zero divergences**; `ms_sig` has r_hat = 54.
Both broken runs still place the truth inside their (meaningless, far too wide) error bars, so a
z-score check against truth does **not** catch this. Watch `num_steps` saturating at 2^10 and r_hat.

## Cost: do not use a GPU

1000 warmup + 1000 draws, 8 chains, same code and same seed:

| N | dims | 8x Xeon 4216 (CPU) | 1x A40 (GPU) |
|---|---|---|---|
| 100 | 307 | 5.2 s (788 ESS/s) | 155 s (23 ESS/s) |
| 1000 | 3007 | **5.2 s (146 ESS/s)** | 493 s (1.7 ESS/s) |

The GPU is ~95x slower at N=1000, and the posteriors agree to three decimals. We run in float64,
which the A40 executes at 1/32 of its float32 rate, on arrays of ~50k elements -- far too small to
amortise the per-leapfrog kernel launches. Chains must also be `vectorized` on one GPU rather than
mapped across devices.

Both CPU rows take 5.2 s because **compile time dominates**; the sampling itself is well under a
second. This is not a compute-bound problem. It was a geometry-bound one (see above).

Do not trust a naive `jax.grad` microbenchmark here: timing `jax.jit(jax.value_and_grad(pot))(z)` in
a Python loop measures dispatch overhead (~0.25 ms/call), not the kernel. Inside NUTS's compiled
`lax.scan` the real gradient is tens of microseconds. That benchmark over-estimated the cost of the
N=1000 run by ~100x.

The cluster is still the right home for the many-seed calibration runs (simulation-based
calibration over ~50 realisations), but as a **CPU array job**, not a GPU job.

## Information content

Per-lens S/N ~ 0.5: individual lenses are undetected and the hierarchy does the inference, as in
SL18. Stacked S/N ~5.8 at N=100, ~21 at N=1000. So N=100 constrains `mh_mu` and `mh_beta` but leaves
`mh_sig` and `c_sig` prior-dominated; all seven only at N=1000.

## Deferrals

- `mstar_cut` selection term dropped (the mock has no cut). When restored, note that
  `examples/infer_nfw_shmr.py:147` compares the raw N(0,1) draw against `mstar_cut = 11.0`
  instead of `mstar_here`; the erf is ~-1 for essentially every sample.
- Fixed source count per lens. Real catalogues are ragged: pad to `(N_lens, S_max)` + mask.
- NFW + point mass only. gNFW (the paper's M/L-gradient result) needs `Sigma(R; gamma)`, which has
  no closed form: replace `wl_profiles/gnfw.py`'s HDF5 grids with Gauss-Legendre quadrature in JAX.
- `ndinterp.py` patched for modern scipy (`scipy.float64`, `scipy.rollaxis` were removed).
