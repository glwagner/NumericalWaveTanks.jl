# Implementation plan: wind-drift profile experiments

This plan supports the research objectives in
[`research_objectives.md`](research_objectives.md). It also builds on the
DNS-snapshot machinery described in [`dns_plan.md`](dns_plan.md), which is
referenced rather than duplicated below.

## Strategy at a glance

Two stages, each producing artifacts the next stage consumes:

```
   Stage A (DNS)          Stage B (extended LES)         post-processing
   ─────────────          ──────────────────────         ───────────────
   experiments/           experiments/                   analysis/
   constant_waves.jl  →   stratified_wave_tank.jl    →   profile/Hovmöller
   wagner_..._les.jl      (or new "long_run" script)     plots, similarity
                                                         fits, Doppler
   Run from t=16 → 18-20s | Restart at t* from saved IC
   small domain (W23)     | DEEPER domain (0.5 m)
   192² × 128             | preserves Δz_min near top
   3D snapshots saved     | runs out to t = 60-180 s
                          | matrix sweeps Laₜ, Ri, Q_b
```

The DNS exists only as an IC factory. We are not chasing an asymptotic
DNS solution; we just need a 3D state that's past the linear-instability
stage so the LES doesn't have to wait through it.

## Stage A — DNS for initial conditions

Already plumbed in `experiments/constant_waves.jl` (see `dns_plan.md`).
Two-step rollout — a low-res functional test, then a high-res production
run that actually resolves the W23 DNS scales.

### A1 — Test (single H100, this cluster)

- 192² × 128 (or smaller, e.g. 128² × 64) with random IC.
- Goal: verify the Checkpointer + 3D fields writer fire, the eigenmode-
  optional branch works, output files load cleanly with `set!`.
- Wall time: a few minutes.
- *Not* a physics-quality DNS — this is a plumbing test only.

### A2 — Production DNS (multi-GPU, DeltaAI / GH200)

- **Resolution: 768² × 512** (matches the W23 DNS).
- Domain: 0.2 × 0.2 × 0.1 m, vertically stretched (refined near surface).
- Wind, wave, and Stokes-drift parameters identical to W23
  (α = 1.2e-5, ε = 0.1, λ = 3 cm).
- Initial condition: laminar Ekman-Stokes profile at t₀ = 16 s plus
  random noise, OR with eigenmode perturbation if the corresponding
  `linear_instability_analysis/*.jld2` is computed at the matching
  resolution.
- Stop time: **t = 18 s** initially (saves ~half the cost vs t = 20 s
  while still containing the jets/plumes regime). Extend to 20 or 22 s
  if the chosen `t*` ends up needing more headroom.
- Snapshots every 0.5 s, both as `JLD2Writer(..., array_type=Float32,
  with_halos=false)` and as full Checkpointer files.

#### Cost on multi-GPU

Single-H100 throughput estimate at 768² × 512 with Centered(2) advection
is ~1 s/iter. With Δt ≈ 0.9 ms set by CFL on the smallest cell, the
iteration counts and wall-time estimates are:

| Window       | Iters  | 1 H100 wall  | 4× GH200 wall (est.) |
|--------------|--------|--------------|----------------------|
| t = 16 → 18 s | ~2700  | ~45 min      | ~15 min              |
| t = 16 → 20 s | ~5000  | ~80 min      | ~25 min              |
| t = 16 → 22 s | ~7000  | ~115 min     | ~40 min              |

The 4-GPU number assumes ~3× scaling — limited by the all-to-all
communication in the FFT-based pressure solver. Better than 3× would
be a pleasant surprise; worse is possible if internode bandwidth is
the bottleneck. Confirm with a 1-iteration benchmark before committing
to the full run.

#### Multi-GPU setup

The relevant Oceananigans pieces:

```julia
using MPI; MPI.Init()
using Oceananigans.DistributedComputations: Distributed, Partition

arch = Distributed(GPU(); partition = Partition(2, 2, 1))
grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), ...)
model = NonhydrostaticModel(grid; ...)
```

Launched via `mpirun -n 4 julia --project ...` or Slurm's `srun -n 4 ...`.

#### Output combining

Each rank writes its own JLD2 (`..._rank0.jld2`, ...). Since the LES
restart workflow expects a single file, we combine eagerly: a small
`analysis/combine_dns_snapshots.jl` runs after the DNS, reads each rank
file's interior, places it in the global array using the partition
metadata embedded by Oceananigans v0.107+, and writes a merged
`..._merged_t<...>.jld2`. The merged file is what the LES ingests via
`set!(les_model, ...)`.

### A3 — Snapshot selection

Inspect `xy_top` and `xz_left` visualizations of each saved snapshot;
pick `t*` matching jets-formed, pre-isotropic. Best guess: `t* ∈
[18, 19] s`. Commit a small file (e.g. `experiments/dns_t_star.txt`)
naming the chosen iteration so Stage B is reproducible.

Outputs of Stage A: one `..._3d_fields.jld2` snapshot per 0.5 s in
`(u, v, w, c)`, plus full Checkpointer files for restart, plus the
merged single-file IC at `t*`.

## Stage B — Extended LES from DNS state

### Domain extension and IC strategy

The DNS domain is 0.2 × 0.2 × 0.1 m. The LES domain is **0.2 × 0.2 × 0.5 m**
— same horizontal footprint and resolution, **5× deeper**. Because the
laminar wind-drift boundary layer thickness at t=20 s is `h = √(2νt) ≈ 6.3 mm`,
the DNS bottom (`z = −0.1 m`) is already in essentially quiescent water,
and the Stokes drift `uˢ = 2a²k²ω exp(2kz)` decays as `exp(2k·−0.1) ≈ 10⁻¹⁹`
at the same depth. So the IC is continuous if we just place the DNS state
in the upper 10 cm and zero below:

```
LES IC at t = t*:

z ∈ [−0.1, 0]  m  →  copy DNS u, v, w, c snapshot at t* (same Δx, Δy, Δz_top)
z ∈ [−0.5, −0.1] m →  u = v = w = 0, c = 0 (quiescent)
```

A short z-taper across the patch boundary (~5 cells) can suppress any
small discontinuity. In practice the laminar field is already
essentially zero by `z = −0.1 m`, so the taper may be unnecessary; we'll
verify by inspecting the first-second response after IC injection.

The horizontal field is reused as-is — same Lx, Ly, Nx, Ny — so periodic
boundaries remain consistent. There is **no horizontal tiling** in this
plan (the DNS domain is already periodic with period Lx).

### Vertical resolution

Preserve Δz_min ≈ 0.5 mm at the surface to keep the wind-drift boundary
layer well-resolved. With Lz = 0.5 m and the same stretched-z function
(`refinement = 1.5`, `stretching = 8`), reach a target Δz_max around
5–10 mm at the bottom. Two reasonable points in the trade-off:

| Nz   | Δz_min  | Δz_max  | Notes                                    |
|------|---------|---------|------------------------------------------|
| 192  | 0.7 mm  | 12 mm   | cheapest; deep cells are ≈ 1.5 cm wide   |
| 256  | 0.5 mm  | 8 mm    | matches DNS at top; preferred default    |
| 320  | 0.4 mm  | 6 mm    | finer; only if 256 turns out under-res   |

Default for the matrix: **192 × 192 × 256**.

### Numerics

Same as the existing LES script (`experiments/wagner_et_al_2023_les.jl`):

- WENO advection. Currently `WENO(order=9)`; this is overkill for
  implicit LES — `WENO(order=5)` is standard and ~2-3× faster. Switch
  before the matrix sweep.
- `RungeKutta3` timestepper, `cfl = 0.7`, no explicit subgrid closure.
- Molecular ν, κ via `ScalarDiffusivity` (kept as-is per
  earlier discussion).
- Stokes drift: `UniformStokesDrift` with `ConstantStokesShear` callable
  for ε, k.
- Wind stress: `τ = α√t` with `α = 1.2e-5 m²/s^{3/2}` baseline.

### Run matrix

Six runs, organized by what they vary:

| Run  | Phase     | ε    | Stratification | zₕ      | Q_b           | Notes                            |
|------|-----------|------|----------------|---------|---------------|----------------------------------|
| L0   | unstrat   | 0.1  | none           | —       | 0             | baseline; reproduces W23 LES     |
| L1   | unstrat   | 0.05 | none           | —       | 0             | weaker waves → higher Laₜ        |
| L2   | unstrat   | 0.2  | none           | —       | 0             | stronger waves → lower Laₜ       |
| L3   | strat     | 0.1  | yes            | −0.05 m | 0             | mid-depth thermocline            |
| L4   | strat     | 0.1  | yes            | −0.10 m | 0             | deeper thermocline               |
| L5   | strat+B_q | 0.1  | yes            | −0.05 m | −1e-7 W kg⁻¹  | surface buoyancy loss (cooling)  |

Stop time: **`t = 180 s`** (reach quasi-steady with comfortable margin
for time-averaging in the analysis step). Reduce to 60 s if cost is a
problem.

Estimated wall time per run at 192 × 192 × 256 with WENO(5): ~30 min on
one H100. Matrix total ≈ 3 hours.

## Diagnostics

Profile output every 0.2 s, accumulated as time series. For all runs:

- `U(z, t)`, `V(z, t)` — `Field(Average(u, dims=(1,2)))` etc.
- `⟨u′w′⟩(z, t)`, `⟨v′w′⟩(z, t)` — `Average((u - U) * w, dims=(1,2))`
  computed via `AbstractOperations`.
- `⟨w′²⟩(z, t)` — vertical-velocity variance (a primary target).
- `TKE(z, t) = ½ ⟨u′² + v′² + w′²⟩`.
- Stokes drift shear `∂z uˢ(z)` (analytic, saved once for plotting).
- **Pseudovorticity number** `Ω(z, t)` (see below).

For stratified runs additionally:

- `b(z, t)` — mean buoyancy.
- `⟨w′b′⟩(z, t)` — turbulent buoyancy flux.
- `Ri_g(z, t) = N²(z, t) / (∂z U)² (z, t)` — gradient Richardson number.
- `h(t)` — mixed-layer depth diagnosed as the depth where
  `b(z) − b(0) = α_b · Δb_max` for some threshold `α_b ≈ 0.1`.

### Pseudovorticity number

Defined as

```
Ω(z, t) = ∂z u_L / ∂z u_S
```

with `u_L = U + uˢ` (Lagrangian-mean Eulerian + Stokes) and
`u_S = uˢ`. Equivalently:

```
Ω = 1 + ∂z U / ∂z u_S
```

Interpretation:

- `Ω → 1` means the Eulerian shear is negligible; the Lagrangian shear
  is purely Stokes (pure-wave-driven regime, no turbulent
  redistribution).
- `Ω ≫ 1` means Eulerian shear adds to Stokes shear (laminar
  wind-drift regime, before instability).
- `Ω → 0` (or negative) means Eulerian shear opposes Stokes shear
  (well-mixed regime, jet-like surface profile).

We expect `Ω` to start near `Ω ≫ 1` (laminar) and asymptote to a
specific value characteristic of CL-vortex-force-saturated turbulence.
The depth- and time-structure of `Ω` is the cleanest single signal of
the regime structure of each run.

Computed at every save time and stored alongside the other profiles.

### How profiles are saved

Use a single `JLD2Writer` per run with a NamedTuple of `Field`-typed
profile diagnostics. Reduction-of-products is built via
`Oceananigans.AbstractOperations`:

```julia
using Oceananigans.AbstractOperations: Average

up = u - Field(Average(u, dims=(1, 2)))   # u'
wp = w - Field(Average(w, dims=(1, 2)))
uw = Field(Average(up * wp, dims=(1, 2))) # ⟨u'w'⟩
```

(Field-of-Average pattern; this is the only place we need to be a
little careful about how Oceananigans handles `Field(Average(...))` vs.
in-place buffers.)

## Phasing

1. **Phase 1 — DNS plumbing & smoke (current PR / branch).** Already
   done: optional eigenmode IC, Checkpointer + 3D fields writer, plan
   docs in `experiments/`. Remaining: actually run a small DNS test
   (Stage A1) end-to-end.

2. **Phase 2 — DNS production (next PR).** Run Stage A2 at 192² × 128.
   Inspect snapshots; pick `t*`. Commit a chosen-snapshot pointer (a
   small file referencing the IC checkpoint path) so Stage B is
   reproducible.

3. **Phase 3 — LES IC machinery (next PR).** New script
   `experiments/extended_les_from_dns.jl` (or a refactor of
   `wagner_et_al_2023_les.jl`) that:
   - constructs the deep-domain LES grid;
   - loads the DNS snapshot via `set!(les_model, "..._3d_fields.jld2"; iteration=...)`;
   - zero-pads below `z = −0.1 m`;
   - applies optional taper near `z = −0.1 m`;
   - sets up the augmented profile diagnostics.

4. **Phase 4 — Unstratified matrix (next PR).** Run L0–L2. Plot U(z, t),
   ⟨w′²⟩(z, t), Ω(z, t) Hovmöller; first-pass log-law fit on the
   quasi-steady tail.

5. **Phase 5 — Stratified matrix (next PR).** Run L3–L5. Add
   buoyancy-related diagnostics. Repeat the similarity-fit analysis.

6. **Phase 6 — Synthetic Doppler (separate analysis script).** Take the
   saved `U(z, t)` profiles, evaluate `c̃(k) = ∫ U(z) Q(z, k) dz`, save
   as JLD2 / CSV for downstream PEDM/EKI testing. No changes to
   experiment scripts.

Each phase merges to `main` before the next starts, so we never have
multiple long branches in flight.

## Cost rough-out

Rough wall-time budget on a single H100:

| Phase | Resolution      | Runs | Wall / run | Phase wall |
|-------|-----------------|------|------------|------------|
| 2     | 192² × 128      | 1    | ~3 min     | ~3 min     |
| 4     | 192² × 256      | 3    | ~30 min    | ~90 min    |
| 5     | 192² × 256      | 3    | ~30 min    | ~90 min    |

Total simulation time: well under 4 hours of GPU. The harder cost is
disk: at 192² × 256 each profile timeseries file accrues
~ 4 fields · 4 bytes · 256 · 900 saves = 4 MB; XY/XZ slice files
each ~10 MB; total per run ~ 50 MB of analysis-relevant output.
Comfortable.

## Open questions

- **Bottom boundary condition for the deep domain.** Currently `Bounded`
  in z with default no-flux on tracers and a free-slip-equivalent on
  velocities. With 0.5 m depth and the wind-driven layer reaching
  ~ 10 cm, the bottom should never feel the surface forcing in
  a 60–180 s window — verify by checking that the lowest 10 cm stays
  numerically zero in the baseline.
- **IC injection transient.** Does the abrupt transition from "DNS
  state" to "zero" at `z = −0.1 m` excite a transient internal wave?
  If yes, a few-cell taper or `tanh(z; z_0=-0.1, δ=0.005)` mask on the
  IC will fix it. We'll know from the first-second movie.
- **WENO order revisit.** Currently 9; switching to 5 is a 2–3×
  speedup with no expected loss in implicit-LES quality. Make this
  switch the same PR as Phase 3.
- **Single-snapshot vs. ensemble IC.** One DNS realization gives one
  ensemble member. If statistics aren't stationary enough we may need
  to re-launch the matrix from a second `t*` snapshot and ensemble-
  average post-hoc.
- **Pseudovorticity number sign convention.** The definition above
  treats `∂z u_S` as positive (which it is — Stokes shear is positive
  upward, with our `b = -gρ/ρ₀` and `z` upward sign convention). Worth
  documenting in the analysis script header so plot signs are
  unambiguous.
- **Confirming W23 transition timing.** Once the DNS snapshots are in
  hand, plot peak `max|w|(t)` and `⟨w′²⟩(t)` across the snapshot
  library to mark the actual transition window — don't just trust the
  coarse-LES timing extrapolation.

## References

- Wagner, Pizzo, Lenain, Veron 2023, "Transition to turbulence in
  wind-drift layers", JFM 976 A8.
  [arxiv:2307.15291](https://arxiv.org/abs/2307.15291)
- Teixeira 2018, "A model for the wind-driven current profile in the
  ocean surface mixed layer".
- Large, McWilliams, Patton 2019, "Asymptotic similarity in the
  surface boundary layer with Langmuir turbulence".
- Sullivan, McWilliams, Melville 2004 (breaking-wave LES — the buoyancy
  template script in `experiments/sullivan_mcwilliams_melville_2004.jl`).
- Smeltzer, Esoy, Adnan, Ellingsen 2019, "An improved method for
  estimating sub-surface vertical current profiles using surface waves"
  (PEDM).
- Stewart, Joy 1974, "HF radio measurements of surface currents".
- Craig, Banner 1994 (near-surface dissipation scaling).
