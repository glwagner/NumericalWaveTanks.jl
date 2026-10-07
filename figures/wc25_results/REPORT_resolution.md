> **Resolution status: all 256² and 256³ simulations are PROVISIONAL — VERY LOW RESOLUTION. Coarser runs are provisional as well. Higher-resolution runs are checks, not evidence of convergence by themselves; physical conclusions remain provisional until supported by resolution comparisons.**

# WC25 resolution follow-up — report (INTERIM — 512³ runs and the 256³ r = 0.3 pair are PENDING; their sections will be added after review)

Campaign of 2026-10-03 (assignment `/u/glwagner/wc25_resolution_coordination/assignment.txt`). Notation: u′_rms,0 is the
initial rms fluctuation velocity, u′_rms(t) = √(2K_fluct) the instantaneous one; q is reserved for PV. Branch
`glw/wc25-resolution`, data root `/work/hdd/bhcr/glwagner/wc25_resolution_2026-10-03`.

## 1. Matched initial conditions (a correction to the earlier comparisons)
- The original IC generator draws its random coefficients on an N-dependent array, so the "seed 1" 128³ and 256³ ICs are
  **different realizations** (low-mode coefficient correlation ≈ 0.003). Every 128³-vs-256³ comparison in the first
  campaign therefore mixed resolution with realization variability. Those results are preserved and labelled as such.
- `embed_ic.jl` reconstructs the 256³ draw exactly (legacy IC reproduced to 2.8e−17) and samples the same band-limited
  continuous field on 128³ and 512³: raw low-mode coefficients agree to 3e−16; after each grid's discrete pressure
  projection and rescaling to u′_rms,0 = 0.03 they agree to 2e−3 (128³) and 5e−4 (512³). Divergence ≤ 1e−14 and w = 0 at
  both walls on every grid. A first attempt sampled each grid half a cell apart (7–14 % coefficient differences); it was
  caught by the verification and corrected before any production run.

## 2. Surface stress, matched three-grid comparison (128³ and 256³ done; 512³ PENDING)
Interim (job 3301821):
| quantity | 128³ matched | 256³ | 128³ legacy s1 | 128³ legacy s2 |
|---|---|---|---|---|
| r = 0.1 onset (depth-mean fluctuation ratio > 1.3) | t = 84 (u*/u′_rms 1.02) | 156 (1.06) | 82 (1.02) | — |
| r = 0.03 onset | 273 (0.71) | 270 (0.66) | 277 (0.74) | 273 (0.73) |
| r = 0 K_jet/K_rel at t = 400 | 0.15413 | 0.13376 | 0.18875 | 0.19761 |
- The r = 0.1 onset-time shift between 128³ and 256³ is a **resolution** effect (same field), not realization spread
  (≈ 2 time units). The onset ratio u*/u′_rms(t) ≈ 1 is nearly grid-independent. Whether the onset time converges needs 512³.
- At r = 0.03 the onset is robust across grids and realizations with the fluctuation classifiers (definitions below).
  The raw volume ratio R_raw crosses 1.3 erratically (t = 514 at 256³, 852 at 128³ legacy seed 1, never for the other two)
  and is not used as a roll classifier.

**Classifier definitions (audited against `analysis/wc25_surface_stress/common.jl`, `resolution.jl` and the saved
variables).** Standard curl signs, ω_x = ∂y w − ∂z v (along-wave axis), ω_y = ∂z u − ∂x w (cross-wave axis).
- R_raw = ⟨ω_x²⟩_V / ⟨ω_y²⟩_V: ratio of the volume means of the *total* squared vorticity components, from the scalar
  statistics file (`ωx²`, `ωy²`: Oceananigans volume `Average` of the face-located kernels ω_x² at (C, F, F) and ω_y² at
  (F, C, F)), interpolated to the profile times. It contains the mean-shear contributions (∂z V)² and (∂z U)²; the wind
  layer's (∂z U)² inflates the denominator, so R_raw stays near 1 even when rolls are present.
- R_fl (the classifier used for all onset times): profiles ⟨ω_x²⟩_xy(z) and ⟨ω_y²⟩_xy(z) on the N + 1 z-faces (saved
  every 1 time unit) minus the mean-shear terms (∂z V)² and (∂z U)² computed from the saved mean profiles on the same
  faces (zero at the walls). The subtraction is exact because ⟨∂z u′⟩_xy = ∂z⟨u′⟩_xy = 0 and ⟨∂x w⟩_xy = 0. Order:
  horizontal mean → subtract mean shear → unweighted mean over all N + 1 faces (walls included) → ratio of the two depth
  means. R_layer: the same with the depth mean restricted to faces with z > 0.75 (the upper quarter of the domain).
- Onset: first profile time t > 5 with the ratio > 1.3.

**Correction to the first report.** `REPORT.md`/`REPORT_surface_stress.md` state that the 256³ r = 0.03 run has "a slightly
earlier roll onset (⟨ω′ₓ²⟩/⟨ω′ᵧ²⟩ ≈ 1.2 at t = 400 in the raw measure)". The primes are wrong: the 1.2 is R_raw (total
vorticity including mean shear; 1.175 at 256³ vs 1.003 at 128³ legacy seed 1 at t = 400). With the fluctuation ratio the
two grids agree: R_fl(400) = 1.658 (256³) and 1.757 (128³ legacy seed 1), onset t = 270 and 277. The apparent
"disagreement at 256³" was a comparison of R_raw with a fluctuation classifier, not a resolution effect. [Update 2026-10-06: at 512³, from the same field as the 256³ run, the R_fl onset is later, t = 343 (273 and 270 at the matched 128³ and 256³), at a similar stress ratio u*/u′_rms = 0.72 (0.71, 0.66); R_fl(400) = 1.387 at 512³. The onset time is resolution dependent and not converged.]
- The earlier ~30 % change of the zero-stress jet fraction between grids (legacy 128³ seed 1 → 256³: 0.18875 → 0.13376,
  −29.1 %) mixed realization and resolution: on the same field the change is 0.15413 → 0.13376 (−13.2 %), while the 128³
  realizations alone span 0.15413–0.19761 at t = 400 (single times; no ensemble at 256³).
- 256³ r = 0.03 extended to t = 1000: K_jet/K_rel 0.92, roll ratio 2.6 (128³ legacy: 0.91, 3.7) — the Langmuir state
  persists at resolution.

## 3. Strong stress and frame sensitivity
- 256³ r = 0.3 deep and matched wave-free control: PENDING.
- Accelerating-frame check (128³, r = 0.3 and 1, one realization each; frame moving with U_b = τt, body force −τx̂,
  momentum carried analytically): the moving-frame bulk velocity stays at 1e−16. Two different kinds of agreement:
  (i) integrated budget — cumulative numerical dissipation to t = 400 agrees with the lab frame to −0.39 % (r = 0.3) and
  +1.0 % (r = 1); (ii) instantaneous statistics — frame-invariant quantities agree to ~1 % until t ≈ 100 and then
  separate (K_rel at t = 400: +7.2 % and +49 %). Late-time window statistics, time mean ± temporal standard deviation over 200 ≤ t ≤ 400 (serially correlated samples: the std describes temporal
  variability, not a sampling error):

  | r | quantity | lab frame | moving frame | difference |
  |---|---|---|---|---|
  | 0.3 | K_rel/K₀ | 0.5205 ± 0.0534 | 0.5747 ± 0.0589 | +10.4 % (≈ 1 std) |
  | 0.3 | K_jet/K_rel | 0.384 ± 0.100 | 0.407 ± 0.156 | within spread |
  | 0.3 | R_fl | 1.1296 ± 0.0180 | 1.0782 ± 0.0278 | −0.051 (≈ 2 std) |
  | 0.3 | R_layer | 1.1529 ± 0.0289 | 1.1063 ± 0.0452 | −0.047 |
  | 0.3 | mean ε/K₀ | 1.5472e−2 | 1.5256e−2 | −1.4 % |
  | 1 | K_rel/K₀ | 4.509 ± 0.416 | 4.961 ± 0.384 | +10.0 % (≈ 1 std) |
  | 1 | K_jet/K_rel | 0.357 ± 0.072 | 0.412 ± 0.046 | +0.055 (≈ 1 std) |
  | 1 | R_fl | 1.1052 ± 0.0201 | 1.0436 ± 0.0162 | −0.062 (≈ 3 std) |
  | 1 | R_layer | 1.0625 ± 0.0259 | 1.0306 ± 0.0237 | −0.032 |
  | 1 | mean ε/K₀ | 0.5542 | 0.5581 | +0.7 % |

  The relative energy is ~10 % higher and the roll ratio 0.03–0.06 lower in the moving frame at both stresses. The
  consistent sign suggests a modest frame (Galilean) sensitivity of the WENO discretization under strong mean advection,
  of order 10 % in K_rel; with one realization per case and correlated samples it is not established as significant.
  The instantaneous separation after t ≈ 100 is consistent with sensitivity to small perturbations but two runs do not
  prove chaos, and the same-sign window differences above mean that not all of it can be attributed to chaotic
  decorrelation. Roll ratios stay ≤ 1.2 in both frames, so the absence of wave-organized rolls at
  r = 0.3 and 1 in this setup (128³, one seed, t ≤ 400) does not depend on the frame; it is not a universal no-roll
  condition.

## 4. Two-dimensional ring-9 baseline at 256², 512², 1024² (same coefficients; job 3301470, analysis res3)
At 200 T_e:
| family | energy drift K/K₀ − 1 (256 / 512 / 1024) | K_jet/K | anisotropy | centroid (cycles) |
|---|---|---|---|---|
| shallow | +2.8e−4 / +7.8e−5 / −3.0e−5 | 0.050 / 0.050 / 0.045 | 0.70 / 0.71 / 0.69 | 5.09 / 5.14 / 5.18 |
| deep | +6.2e−3 / +9.7e−4 / +1.2e−4 | 0.39 / 0.38 / 0.45 | 0.62 / 0.59 / 0.52 | 4.12 / 4.06 / 4.32 |
| none | −5.9e−3 / −2.5e−3 / −6.7e−4 | 0.12 / 0.13 / 0.15 | 0.01 / −0.02 / −0.04 | 2.60 / 2.38 / 2.70 |
- The deep-wave energy drift at 200 T_e falls by 6.4× and 8.0× per doubling (+6.16e−3 → +9.66e−4 → +1.21e−4): energy
  conservation of the discrete Stokes terms improves with resolution. This is evidence of improved energy conservation,
  not proof that the dynamics have converged.
- Shallow waves: jet fraction 0.0499 / 0.0497 / 0.0453, anisotropy 0.696 / 0.715 / 0.688, centroid 5.09 / 5.14 / 5.18
  cycles — within 10 % across the three grids (largest change: jet fraction −8.9 % from 512² to 1024²).
- Deep waves: the jet fraction changes 0.3911 → 0.3839 (−1.8 %) from 256² to 512² but 0.3839 → 0.4467 (+16.4 %) from 512²
  to 1024², and the anisotropy 0.620 → 0.591 → 0.521 (−11.9 % for the second doubling). The close 256²/512² agreement was
  therefore not convergence; the deep-wave jet statistics remain resolution-sensitive at the 10–20 % level (and are
  seed-sensitive, see the first report). Single realizations at one time; no ensemble spread is available at 1024².
- Wave-free: dissipation decreases with resolution as expected; statistics of a decaying 2-D condensate with few
  vortices are dominated by chaotic variability (jet fraction from random projection).

## 5. Budget
See `/u/glwagner/wc25_resolution_status.md` (ledger, billed and reserved GPU-hours).

## 6. Two-dimensional k₀ = 12, rb = 2 ring ensemble at 256², 512², 1024² (completed 2026-10-04; PROVISIONAL: 256² very low resolution; 512²/1024² convergence not established)
Matched continuous Fourier coefficients per seed (shallow seeds 1–6, wave-free seeds 1–3; 60 T_e; same physical parameters as the
published 256² ensemble). IC preflight (job 3304355), run after the 1024² throughput pilot (shallow seed 1, job 3304346) and before the ensemble job: stored coefficients agree with the 256² IC to
≤ 3.1e−16 after one uniform factor, coefficients re-extracted from the grid streamfunction to ≤ 4.1e−16, energy outside the drawn mode
box zero to round-off; the uniform energy-normalization factor (rescaling to the same measured u′_rms,0 = 4.397621e−05) is
λ = 0.99791–0.99809 at 512² and 0.99737–0.99762 at 1024² (report `initial_conditions/preflight_k12_rb2_report.txt`). All 18 final cases
validated COMPLETED at 60.000 T_e before analysis. Figures `figures/wc25_vallis2d/ens12_{spectra_shallow_heatmap,spectra_none_heatmap,
statistics,flux}.png` (no overlays), numbers `ens12_metrics.txt` (script `analysis/wc25_vallis2d/ensemble_resolution.jl`).

Ensemble mean ± standard deviation over seeds at 60 T_e (shallow, 6 seeds):
| quantity | 256² | 512² | 1024² |
|---|---|---|---|
| anisotropy | 0.692 ± 0.026 | 0.675 ± 0.023 | 0.668 ± 0.024 |
| exact-zonal fraction | 0.0119 ± 0.0026 | 0.0113 ± 0.0026 | 0.0110 ± 0.0024 |
| energy centroid (cycles/box) | 9.78 ± 0.15 | 9.96 ± 0.16 | 10.02 ± 0.16 |
| mean over members of each member's peak inverse flux max_{K<10}(−Π)·T_e/K₀ | 0.0103 ± 0.0036 | 0.0102 ± 0.0040 | 0.0102 ± 0.0035 |
| energy drift K/K₀ − 1 | +4.29e−3 | +1.37e−3 | +3.83e−4 |
| enstrophy Z/Z₀ | 0.810 ± 0.015 | 0.883 ± 0.013 | 0.928 ± 0.010 |
| energy fraction at K > 30 cycles | 8.3e−3 | 1.11e−2 | 1.20e−2 |
At 20 T_e, for shallow waves, the selected statistics (anisotropy, exact-zonal fraction, energy centroid, member-peak inverse flux) differ between grids by less than one seed standard deviation, while energy drift and enstrophy do not; the high-k fraction rises 4.6e−3 / 5.1e−3 / 5.2e−3. For the wave-free runs the centroid already differs at 20 T_e (8.57 / 8.68 / 8.73 cycles; seed std ≤ 0.05).
Wave-free (3 seeds) at 60 T_e: centroid 5.40 / 5.37 / 5.44 cycles, exact-zonal fraction 0.057 ± 0.030 / 0.074 ± 0.020 / 0.068 ± 0.034,
energy drift −6.96e−3 / −2.17e−3 / −6.71e−4, enstrophy 0.32 / 0.36 / 0.41.

Interpretation (provisional, descriptive): anisotropy, exact-zonal fraction and member-peak inverse flux change between grids by less than
one seed standard deviation, and their 512² → 1024² change is smaller than 256² → 512². The energy centroid shifts 9.78 → 9.96 → 10.02
cycles, i.e. by more than one seed standard deviation (0.15) over the full range (0.18 then 0.06). The seed standard deviation describes
the spread of individual members; it is not the uncertainty of the paired resolution difference, and no statistical test is implied.
These comparisons are consistent with, but do not demonstrate, convergence of the selected statistics. Small-scale quantities are clearly resolution dependent: enstrophy retained at 60 T_e rises
from 0.81 to 0.93 and the energy drift falls by a factor 3.1 then 3.6 per doubling (improved energy conservation, not proof of converged
dynamics). The 256² results remain PROVISIONAL — VERY LOW RESOLUTION; 512²/1024² results are provisional, convergence not established. No spectral
boundary is drawn on these figures.

Flux statistics (stated explicitly; values in units of Π·T_e/K₀): (a) the table's "member-peak" column is the mean over members of each
member's maximum of −Π(K) for K < 10 cycles; this is the same definition as the overlay audit's ε, and the numbers agree (audit ε at
60 T_e = 3.303e−14 in model units = 1.030e−2 after multiplying by T_e/K₀ = 301.58/9.6695e−10, the 256² entry here). (b) The peak of the
ENSEMBLE-MEAN flux profile is smaller (shallow, 60 T_e: 7.3e−3 / 7.3e−3 / 7.5e−3 at K ≈ 7–8 cycles; FLUX lines in ens12_metrics.txt)
because member peaks sit at different K. (c) The band average of the ensemble-mean −Π over 2 ≤ K ≤ 10 cycles is +2.5e−3 / +2.6e−3 /
+2.7e−3 at 60 T_e (about a quarter of (a)). The audit's "no clear plateau" finding is a statement about the K-dependence (relative spread
of −Π over the band, computed per member and averaged over members); its 60 T_e entry "mean ≤ 0" means that at least one member had a
non-positive band mean, which makes the member-averaged spread undefined — not that the ensemble-mean band average is ≤ 0. (a), (b) and
(c) are different statistics, not a contradiction.
Cost: pilot 0.071, preflight 0.024, attempt 1 0.070 (MEMBERS truncated by `sbatch --export`; re-ran the pilot case), attempt 2 0.457,
dry-run test 0.007, analysis 0.041 GPU-h.
