# Spectral-boundary overlay audit — Vallis-2D shallow waves (PROVISIONAL — VERY LOW RESOLUTION, 256²)

Completed 2026-10-04 (Slurm job 3304332, 0.02 GPU-h). Script `analysis/wc25_vallis2d/rhines_audit.jl` (branch glw/wc25-resolution,
commit a586ff8), run on the existing saved fields; [raw table](../wc25_vallis2d/audit_rhines_overlays.txt). All results are from 256²
single grids (k₀ = 9, rb = 1 pilot, seed 1; k₀ = 12, rb = 2, six-seed ensemble mean) and are provisional; higher grids are being run.

## What the published dashed curves are
`analysis/wc25_vallis2d/figures.jl` (function `dumbbell`) draws K = √(β|cos θ|/U)/2π in cycles/box with β = 1/2 and U = √(2K(t)), the
total rms speed; the angle is measured from the k_x axis and the wall basis maps k_z = nπ to n/2 cycles. No units, factor-of-2π or
orientation error was found. Because rb = β/(U₀K₀²) and the reported rms speed varies by only about 0.21 %, the k_x-axis radius stays at √rb·k₀:
9.00 cycles (pilot) and 16.95–16.97 cycles (ensemble) at all sampled times. These curves are reference scales, not boundaries
fitted to or measured from the spectra.

## Measured quantities (radii in cycles/box on the k_x axis)
| case | t/T_e | U/U₀ | dashed (global Rhines) radius | Eq. 12.14 radius using max inverse flux | max −Π below ring | ΣT/Σ\|T\| | std/mean of −Π, 2 ≤ K ≤ k₀−2 | energy inside dashed curve |
|---|---|---|---|---|---|---|---|---|
| pilot k₀ = 9 | 5 | 1.0000 | 9.00 | 21.7 | 2.6e−12 | 4e−6 | 1.20 | 0.161 |
| pilot | 40 | 1.0000 | 9.00 | 24.3 | 1.5e−12 | 3e−5 | 1.03 | 0.345 |
| pilot | 200 | 1.0001 | 9.00 | 30.9 | 4.5e−13 | 1e−6 | 1.50 | 0.402 |
| ensemble k₀ = 12 (6 seeds) | 5 | 1.0001 | 16.97 | 52.9 | 3.2e−14 | 9e−6 | 1.15 | 0.655 |
| ensemble | 20 | 1.0006 | 16.97 | 45.6 | 6.5e−14 | 3e−5 | 1.18 | 0.581 |
| ensemble | 60 | 1.0021 | 16.95 | 52.9 | 3.3e−14 | 1e−5 | undefined: ≥ 1 member has band mean of −Π ≤ 0 (see note) | 0.570 |

Note (2026-10-04, clarification): the spread column is computed per member and averaged over members. At 60 T_e at least one member
has a non-positive band mean of −Π over 2 ≤ K ≤ k₀ − 2, so the member-averaged spread is undefined. The ensemble-mean band average is
positive (+2.5e−3 after multiplying flux by T_e/K₀ at 256²; REPORT_resolution.md §6).

## Findings (provisional, for the sampled saved fields only)
1. The nonlinear spectral transfer passes a necessary consistency check, ΣT/Σ|T| ≤ 3×10⁻⁵ (energy-conserving transfer). This does not
   by itself establish that the flux diagnostic is accurate.
2. No clear flux plateau is present below the ring at the sampled times: the relative spread of −Π(K) over 2 ≤ K ≤ k₀ − 2 cycles is
   ≥ 1.0, and at 60 T_e at least one member’s band-mean flux is non-positive. A single transfer rate ε is therefore not justified from these samples,
   and no Vallis Eq. 12.14 curve is drawn. (Using the maximum inverse flux anyway gives radii of 22–53 cycles that enclose 88–97 % of
   the energy, so a maximum alone is not a substitute for an inertial-range ε.)
3. The dashed curves enclose 16–40 % (pilot) and 57–66 % (ensemble) of the energy, so they do not delimit the observed depleted region.
4. A scale-dependent turnover boundary √(K³E(K)) = β/K could not be evaluated reliably from these binned single-grid spectra.
5. The depleted region in the ensemble mean is visibly smaller than the dashed curves. Any boundary drawn for it would have to be an
   explicitly empirical one; none is drawn. These statements concern these samples and grids, not decaying β-turbulence in general.

## Consequences for figures
Existing dashed curves remain as labelled reference scales only; no curve has been resized. Future ensemble and resolution figures
draw no boundary unless a measured quantity supports it.
