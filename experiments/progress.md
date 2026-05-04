# Progress: NumericalWaveTanks W23 reproduction & extensions

State as of branch `glw/les`, commits `0368479..affe784` (after merge of
`glw/update-oceananigans` to `main`).

## What was done

This branch reproduces the Wagner et al. 2023 ("Transition to turbulence
in wind-drift layers", JFM 976 A8) DNS, then builds an LES-restart
pipeline and runs an unstratified ε-sweep + stratified parameter matrix
on top of it. All scripts target a single H100 GPU on this cluster.

### Stage 1 — Oceananigans update (merged to `main`)

Commits `05d9612 → 6defa95`. Bumped the project from Oceananigans
v0.104.2 → v0.107.4 and fixed four breaking changes that surface in
the experiment scripts:

1. `NonhydrostaticModel(; grid, ...)` → `NonhydrostaticModel(grid; ...)`
2. `JLD2OutputWriter` → `JLD2Writer`
3. `FluxBoundaryCondition` with closure-captured variables must use
   `parameters` kwarg (GPU compile constraint)
4. `CenteredSecondOrder()` → `Centered(order=2)`

### Stage 2 — Branch `glw/les`: WENO-LES + stratified scripts

Commits `0368479 → 4e3089d`:

- **`experiments/wagner_et_al_2023_les.jl`** — implicit-LES counterpart
  of `constant_waves.jl`. WENO(order=9) acts as the subgrid model;
  molecular ν, κ retained; random-noise IC.
- **`experiments/stratified_wave_tank.jl`** — buoyancy version with
  two-layer tanh thermocline.
- **`analysis/plot_u_xz_slice.jl`**, **`animate_uw_xz_slice.jl`** —
  visualization scripts.

A bug in the stratified IC (heavy-over-light) was caught and fixed in
`5d4be8e` (`-Δb/2 (1+tanh)` → `-Δb/2 (1-tanh)`).

### Stage 3 — DNS-as-IC pipeline

Commits `8abe3e7 → 13f0481`:

- **`experiments/constant_waves.jl`** retrofitted with optional MPI
  distribution, eigenmode-IC fallback, and a slim 3D-fields output (no
  Checkpointer; Float32; 4 fields). Per-run footprint went from ~280 GB
  to ~32 GB at 768²×512.
- **`analysis/combine_dns_snapshots.jl`** — eager combiner that stitches
  per-rank JLD2 shards into a single merged file using
  `timeseries/<field>/serialized/indices` metadata.
- **`experiments/extended_les_from_dns.jl`** — LES-restart script. Loads
  a DNS snapshot via `FieldTimeSeries`, embeds it in the upper portion
  of a deeper LES domain via `interpolate((x,y,z), field)` per-cell IC
  callback, and zero-fills below.
- **`experiments/{research_objectives,claude_code_plan,dns_plan}.md`** —
  plan documents.

### Stage 4 — Two failed DNS runs, then the right one

Commits `fcdce98 → 13f0481`. The first two production DNS runs used
wrong parameters:

- ε=0.10, L=0.20³ (W'=0.001) — IC noise too small, transition didn't
  develop in the 4-s window.
- ε=0.14, L=0.20³ (W'=0.01) — picked ε from the analysis-script prefix
  `ep140_..._L20_20_10`, which is a *different* DNS variant the
  authors ran, not the paper's headline case.

The actual W23 paper specifies on page 8: **ε = 0.11, U' = 5 cm/s**,
on a **10 × 10 × 5 cm domain** (not 20×20×10). The third DNS run
matched these.

### Stage 5 — W23-correct DNS (`13f0481`)

DNS at 768²×512 in 0.1×0.1×0.05 m, ε=0.11, U'=5 cm/s (script W'=0.005),
t=16→22. Result: clean Langmuir streaks visible at t=18, jets-and-plumes
regime by t=19, 3D-transition by t=20. Matches the paper's stage
diagram (Fig. 1 caption: t=16-18 instability, t=18-20 self-sharpening,
t>20 turbulence). This run only got to t=20 because of Slurm wall-time
limit, but still captured the snapshots needed.

### Stage 6 — LES matrix L0–L5

Commits `5620839 → 49b0dd4`.

LES restart at t=19 from the W23 DNS, deep 0.25 m domain, 192²×128.
Per-run wall time ~10–13 minutes on a single H100.

**Unstratified ε-sweep (L0–L2):**

| Run | ε     | u*_eff (cm/s) | u*_eff / u*_lam |
|-----|-------|---------------|-----------------|
| L1  | 0.055 | 1.01          | 1.05            |
| L0  | 0.11  | 0.84          | 0.87            |
| L2  | 0.22  | 0.28          | 0.29            |

(u*_lam = √(α√t) = 0.96 cm/s at t=60.) **u*_eff decreases with ε** —
qualitatively matches Teixeira (2018) Langmuir-modified similarity
(stronger Langmuir → enhanced mixing → reduced effective wall stress).
The z₀ values from the simple log-law fit are unphysical at high ε
because the upper layer becomes nearly well-mixed and there is no
recognizable log layer.

**Stratified matrix (L3–L5):**

| Run | zₕ      | Q_b (W/kg) | wall  |
|-----|---------|-----------|-------|
| L3  | -0.05 m | 0          | 7 min |
| L4  | -0.10 m | 0          | 14 min|
| L5  | -0.05 m | -1×10⁻⁷    | 7 min |

- Mixed-layer depth follows zₕ (capped at the thermocline).
- Surface buoyancy flux (L5) destabilizes the upper layer, producing
  visibly enhanced ⟨w²⟩ and ⟨w'b'⟩ near the surface.

### Stage 7 — Extra diagnostics

Commits `076a163 → 26ea7e0`.

- **Pseudovorticity Ω(z, t) = ∂z u_L / ∂z u_S** (`plot_pseudovorticity.jl`).
  At lab scale Ω ≫ 1 nearly everywhere — wind-driven Eulerian shear
  dominates Stokes shear; the Ω→1 region is confined to the very-near-
  surface (top mm). Stronger ε grows the Stokes-comparable region,
  consistent with ε² scaling of Stokes shear.
- **Log-law similarity fit** (`loglaw_fit.jl`).
- **Synthetic Doppler observations** c̃(k, t) (`synthetic_doppler.jl`)
  — the depth-weighted Stewart–Joy kernel evaluated on the simulated
  U(z, t). Closes the loop with the PEDM/EKI motivation in
  `research_objectives.md`.

### Stage 8 — Stratified DNS at zₕ=-3 cm

Commits `7ac9572 → affe784`.

- **`experiments/stratified_dns.jl`** — DNS counterpart of
  `stratified_wave_tank.jl` (Centered(2), molecular ν,κ). Same W23
  paper params plus a tanh thermocline; configurable Q_b.
- Submitted at 768²×512 in 0.1×0.1×0.05 m, zₕ=-3 cm, Δρ=35 kg/m³,
  no surface flux, t=16→22. Ran in 1.5 h on gpu-prod-1.
- **Finding**: the thermocline at -3 cm is too deep to be reached by
  wind-driven turbulence in the 6-s DNS window. Surface streak
  dynamics nearly identical to the unstratified case at t=18–22 (very
  subtle suppression of 3D transition at t=20–22). Buoyancy field
  stays sharp at the prescribed depth — visible in
  `dns_compare_strat_vs_unstrat.png`.

To probe stratification effects meaningfully, would need either: a
shallower thermocline (e.g., zₕ=-1 cm), a longer integration via LES
restart (mixed layer grows past -3 cm by t≈40-60 s), or stronger
forcing.

## Files produced

In repo (under `glw/les`):

- `experiments/`: `constant_waves.jl` (slimmed), `extended_les_from_dns.jl`,
  `extended_stratified_les_from_dns.jl`, `stratified_wave_tank.jl`,
  `stratified_dns.jl`, `wagner_et_al_2023_les.jl`, plan markdowns,
  `progress.md` (this file).
- `analysis/`: `plot_u_xz_slice.jl`, `animate_uw_xz_slice.jl`,
  `combine_dns_snapshots.jl`, `plot_dns_surface_u.jl`,
  `plot_les_profiles.jl`, `compare_les_matrix.jl`,
  `compare_strat_les_matrix.jl`, `plot_pseudovorticity.jl`,
  `loglaw_fit.jl`, `synthetic_doppler.jl`,
  `compare_dns_strat_vs_unstrat.jl`.
- `dns_production.batch`, `dns_stratified.batch` — Slurm submission.

In `~/Projects/`:

- `dns_W23_*` — DNS surface streaks + xz animation
- `dns_strat_*` — stratified DNS at zₕ=-3 cm
- `dns_compare_strat_vs_unstrat.png` — DNS strat-vs-unstrat
- `les_W23_*` — baseline LES restart
- `les_matrix_compare.png` — L0–L2 unstratified ε-sweep
- `les_strat_matrix_compare.png` — L3–L5 stratified
- `pseudovorticity_matrix.png`, `loglaw_fit.png`, `synthetic_doppler.png`
  — extra diagnostics

## Open issues / next steps

1. **LES restart from the stratified DNS** (~30 min compute) to extend
   the stratified case to t=60 and probe the mixed-layer / thermocline
   interaction at long times.
2. **Update `extended_stratified_les_from_dns.jl`** to optionally load
   the buoyancy field from a stratified DNS source, instead of always
   initializing analytically. (Currently the stratified LES uses
   unstratified DNS u/v/w/c + analytic tanh b — fine if the DNS-side b
   is unchanged from IC, but limiting in general.)
3. **Eigenmode IC for distributed grids** — the current eigenmode-
   loading branch in `constant_waves.jl` reads the full-domain
   eigenmode array; for multi-rank distributed runs it would need
   per-rank slicing. Currently disabled (file missing) so the code
   path is dormant.
4. **Multi-GPU production DNS on DeltaAI** — plumbing is ready (MPI
   support, rank-suffixed file names, `combine_dns_snapshots.jl`
   stitcher). The local cluster only has 2 H100s on separate nodes;
   the multi-GPU path is for the GH200 cluster mentioned in
   `research_objectives.md`.
5. **Trim `Lz` further** — wind-driven mixed layer reaches at most
   ~15 cm in this regime, so even Lz=0.25 m has ~10 cm of dead space.
   Could go to Lz=0.15 m for cheaper matrix runs; not done yet.
6. **WENO order revisit** — current LES uses WENO(order=5) which is
   fine; WENO(order=9) would be ~3× slower for marginal SGS-quality
   gain.

## Cluster / cost summary

Total compute used in this branch (rough):

- DNS at 768²×512: ~3 hours (3 attempts × ~1 hour each)
- LES matrix L0–L5: ~1 hour total (6 runs × ~10 min each)
- Smoke tests + LES baseline: ~30 min total
- **Grand total: ~5 hours of single-H100 wall time**

Disk: ~73 GB persistent (40 GB stratified DNS + 32 GB unstratified DNS
+ ~1 GB LES outputs). Two failed DNS dirs (~280 GB) deleted after the
W23-correct one ran.
