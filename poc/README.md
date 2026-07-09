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

**Rung 3.** N=100 lenses, 307 dimensions, 2.6 s, 0 divergences, r_hat = 1.00.

## Parameterisation: mixed, not non-centred

Measured at N=100, 300 warmup + 300 samples, 4 chains:

| | median leapfrogs | saturating tree depth | divergences | min ESS | ESS/s |
|---|---|---|---|---|---|
| fully non-centred | 543 | 50% | 0 | 3 | 1.3 |
| **mixed** | **63** | **0%** | **0** | **512** | **235** |
| fully centred | 127 | 25% | 5 | 3 | 1.2 |

Non-centring only pays when the data are *un*informative about the latent. `logM*` is pinned by
`lmstar_obs` (0.15 dex, against 0.25 dex population scatter), so it wants the **centred** form.
`logM200` and `logc200` are constrained only by weak lensing at per-lens S/N ~ 0.5, so they want
**non-centred**. Getting this wrong costs 180x in ESS/s.

Note the failure mode: the fully non-centred model reports **zero divergences** while delivering an
effective sample size of 3. It does not announce itself. Watch `num_steps` saturating at 2^10.

## Cost

A gradient is 0.25 ms at N=100 (307 dims) and 0.97 ms at N=1000 (3007 dims), on CPU. At ~63
leapfrogs/draw the whole N=1000 problem is minutes on a laptop. This is not a compute-bound problem;
it was a geometry-bound one.

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
