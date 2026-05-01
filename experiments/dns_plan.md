# DNS plan: Wagner et al. 2023 wind-drift transition

## Goal

Run a 3D direct numerical simulation of the [Wagner et al. (2023)][w23]
wind-drift-layer transition problem and save full 3D state snapshots so that
any one of them can be used as an initial condition for an LES restart. The
script of record is [`experiments/constant_waves.jl`](constant_waves.jl). The
purpose of saving is to bypass the slow linear-instability stage in the LES,
which is poorly represented at coarse resolution: a DNS that is well into
the jets-and-plumes stage gives the LES a fully-developed (or near-developed)
3D velocity field with realistic spectral content.

## Physical setup (from `constant_waves.jl`)

- Domain: 0.2 × 0.2 × 0.1 m, vertically stretched (refined near surface)
- Topology: periodic in x, y; bounded in z
- Wave: λ = 3 cm capillary-gravity wave (k = 2π/0.03 m⁻¹), slope ε = 0.1.
  `ω = √(gk + γk³)` with `γ = 7.2e-5 m³/s²`
- Stokes drift: `UniformStokesDrift` with shear `2 a²k²ω exp(2kz)`
- Wind stress: `τ = -α√t`, default `α = 1.2e-5 m²/s^{3/2}`
- Viscosity: `ν = 1.05e-6 m²/s` (water, molecular)
- Tracer: rhodamine-like passive tracer with `κ = 1e-7 m²/s`
- Simulation begins at `t₀ = 16 s` from the analytic Ekman–Stokes profile,
  i.e. the same time at which W23 finds capillary-gravity ripples appearing
  in the laboratory experiments

## Transition stages (W23 narrative)

The flow evolves through four stages between `t = 0` and full 3D turbulence:

1. **Laminar wind-drift layer** — viscous stress accelerates a shear
   profile; ripples have not yet formed. (Not simulated; `t < 16 s`.)
2. **Wave-catalyzed instability** — once ripples are present (modeled via
   `UniformStokesDrift` from `t₀`), the linear instability of the laminar
   profile begins to grow. The dominant unstable modes are computed
   separately by `experiments/linear_instability_analysis/linear_instability.jl`.
3. **Jets and downwelling plumes** — the linear instability sharpens into
   along-wind (`x`-direction) jets at the surface and downward-propagating
   plumes. **This is the regime we want as an LES initial condition.**
4. **Three-dimensional turbulence** — the jets and plumes destabilize and
   the flow becomes fully turbulent.

### Approximate timing

W23 reports run times of about 6 seconds after `t₀ = 16 s` (so to `t ≈ 22 s`)
to capture the transition. Based on the LES baseline run we have already
completed (`experiments/wagner_et_al_2023_les.jl` at 192³, t = 16→22 s):

| Approx. time | Stage                                  |
|--------------|----------------------------------------|
| 16.0–17.5 s  | Laminar; instability invisible by eye  |
| 17.5–19.0 s  | Linear growth (still mostly laminar)   |
| 19.0–20.5 s  | Jets / plumes emerge                   |
| 20.5–22.0 s  | Transition to 3D turbulence            |

These windows are coarse-LES estimates; the DNS will resolve them more
sharply and timings may shift slightly. **The save target is the
plumes-just-formed window, somewhere around `t = 19.5–20.5 s`.**

## Save strategy

`constant_waves.jl` (this branch) writes two complementary 3D outputs:

- `JLD2Writer` → `<prefix>_3d_fields.jld2` — `(u, v, w, c)` fields,
  `Float32`, no halos. This is the right format for re-injecting into an
  LES via `set!(les_model, ...)`.
- `Checkpointer` → `<prefix>_checkpointer_iteration*.jld2` — full bit-exact
  state for either restarting the DNS itself or interpolating to a
  different grid.

Both fire on `TimeInterval(save_interval_3d)`, default 0.5 s. With
`cleanup = false` every snapshot is kept.

### File-size budget

Let `M = Nx · Ny · Nz`. Per-field interior size in Float32 is `4M` bytes;
the full 3D fields file holds 4 fields and grows by `≈ 16M` bytes per save.
The Checkpointer writes more (Float64, with halos and pressures).

| Resolution     | M          | 16M bytes (3D fields/snap) | snaps in 6 s @ 0.5 s | Total fields | Checkpointer per snap (rough) |
|----------------|------------|----------------------------|----------------------|--------------|-------------------------------|
| 192³ (test)    | 4.7 M      | 75 MB                      | 13                   | 1.0 GB       | 0.5 GB                        |
| 384²×256       | 38 M       | 600 MB                     | 13                   | 7.8 GB       | 4 GB                          |
| 768²×512       | 302 M      | 4.8 GB                     | 13                   | 63 GB        | 30 GB                         |

For the **production 768²×512 DNS** the totals are large but tractable:
the user asked for 0.5 s saves; alternatives if disk pressure is a problem
are (a) widen to 1 s outside the transition window, (b) drop the
Checkpointer (keep only the 3D fields), (c) write only `u, v, w` and skip
the tracer.

### Recommended save window

If saving all 13 snapshots is too much, save densely only around the
transition window:

- **t = 16.0–18.5 s**: every 1.0 s (3 snapshots)
- **t = 18.5–21.0 s**: every 0.25 s (10 snapshots)
- **t = 21.0–22.0 s**: every 1.0 s (1 snapshot)

Total: 14 snapshots ≈ 67 GB Float32 fields at 768²×512.

Either schedule is fine; the simple uniform 0.5-s schedule is currently
what the script implements.

## Initial condition for the DNS

The DNS uses the analytic Ekman–Stokes laminar profile at `t₀ = 16 s` plus
a perturbation. The perturbation is a precomputed unstable eigenmode
(loaded from `linear_instability_analysis/linearly_unstable_mode_*.jld2`)
when available; otherwise random noise (`W' randn()`) only.

For the production 768²×512 DNS, the eigenmode IC is preferred — it seeds
the dominant unstable mode cleanly and gives a faster, cleaner approach
to nonlinear saturation. The eigenmode files are produced by running
`experiments/linear_instability_analysis/linear_instability.jl` at the
matching `(t₀, ε, Ny, Nz, Ly, Lz)`.

## LES restart workflow

1. Run the DNS (`constant_waves.jl`) at production resolution. Snapshots
   land in `<prefix>_3d_fields.jld2` (and Checkpointer files).
2. Pick a snapshot time `t*` corresponding to the desired stage (default
   target: `t* ≈ 20 s`, plumes formed but pre-3D-turbulence).
3. In the LES script, after constructing the LES model, load
   `(u, v, w, c)` from the snapshot and `set!` the LES fields. The LES
   grid will typically be 2–4× coarser than the DNS grid; Oceananigans'
   `set!(model, "snapshot.jld2")` interpolates between resolutions
   automatically when the field grids match topologically (same domain,
   same topology, same z-stretching pattern).
4. Continue the LES from `t*` to the desired stop time.

If the LES grid uses a different vertical stretching from the DNS, an
explicit interpolation step (e.g., via `regrid!` or manual interpolation)
will be required.

## Test resolution

To validate the script changes we run a small DNS at 128²×64
(`Δx ≈ 1.5 mm`, far from a true DNS but enough to exercise every branch).
Random IC, `t = 16 → 17 s`, ~2 snapshots. Sanity checks:

- Both `_3d_fields.jld2` and `_checkpointer_iteration*.jld2` files appear
- A second invocation can `set!` from the saved snapshot without error
- No NaNs, velocities `O(mm/s)`

## Production resolution & cost

W23's DNS is 768²×512 in 0.2³ × 0.1 m (or 0.1²×0.05 in the half-domain
variant). On a single H100 we expect roughly:

- Per-iteration work scales `~M log M` (FFT pressure solve dominates) or
  `~M` (advection); Centered(2) advection is much cheaper than WENO(9).
- The 192³ LES with WENO(9) was ~60 ms/iter.
- The 768²×512 DNS with Centered(2) is ~roughly 60 × (768²·512 / 192²·128) ×
  (Centered/WENO speedup) = ~60 × 64 × 0.3 ≈ 1100 ms/iter.
- 6 s of physics at Δt ≈ 0.5–1 ms ≈ 8000 iterations ≈ 2.5 hours.

That's an estimate, not a guarantee. The first benchmark run will pin
this down.

## Open questions / TODOs

- **Confirm transition timing from W23 figures.** The 17.5–22 s windows
  above are coarse-LES estimates; reading the paper's spectra/energy
  plots will tighten them. (The full-text PDF was not accessible during
  this planning step.)
- **Compatibility of LES and DNS grids.** The current LES uses
  `WENO(order=9)` with halo 5 and 192³ on the same domain as the DNS at
  768²×512. Either the LES grid must coarsen the DNS field exactly
  (factor-of-4 in each direction works cleanly with the existing
  `RectilinearGrid` stretched-z definition) or we need an interpolation
  step.
- **Stratified DNS variant.** Once the unstratified DNS-to-LES path
  works, the same machinery applies to the stratified case. The
  `experiments/stratified_wave_tank.jl` LES script can be similarly
  retrofitted with checkpointer/3D output, and a stratified DNS variant
  can be derived from `constant_waves.jl` by adding `BuoyancyTracer()`
  and a `b` initial profile, mirroring what was already done in the LES.
- **Eigenmode generation at higher resolution.** The available
  `linear_instability.jl` script computes 2D eigenmodes at one
  resolution; for the production DNS we'll need to (a) confirm the
  resolution match, or (b) interpolate the eigenmode onto the DNS grid.

[w23]: https://doi.org/10.1017/jfm.2023.920
