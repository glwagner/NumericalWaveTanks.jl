# Anti-Stokes campaign: how the data were made, what they are for, and how to reproduce or extend them

This document is written for an agent (Claude Code or similar) picking up the campaign. It
records the goals, the physical and numerical design, every step that produced the data in
`/work/hdd/bhcr/glwagner/anti_stokes`, the exact commands, the analysis and plotting recipes,
and the pitfalls that cost time. The short user-facing overview is `README.md`; the pull
request is <https://github.com/glwagner/NumericalWaveTanks.jl/pull/7>.

## 1. Goals

Reproduce, with a wave-averaged (Craik–Leibovich) LES in Oceananigans, the laboratory
measurements of Ellingsen, Rømcke, Smeltzer, Teixeira, van den Bremer, Moen and Hearst,
*Turbulence-induced anti-Stokes flow: experiments and theory*, J. Fluid Mech. 1029, A6 (2026),
doi:10.1017/jfm.2026.11163:

1. **Experiment 1 (wave groups), cases 1.A–1.D.** A Gaussian wave group passes over decaying
   grid turbulence in a current. After the group has passed, the Eulerian mean velocity near the
   surface is reduced by O(10 mm/s): the "anti-Stokes" flow. Targets: sign, magnitude, vertical
   scale k₀|z| = O(1), near-zero depth-integrated transport, driven by −∂z⟨u'w'⟩, and its
   dependence on turbulence intensity, turbulence scale and wave steepness (the 1.C.1/1.C.2 pair).
2. **Does the Lagrangian mean homogenize beneath the packet?** Quasi-equilibrium theory gives
   ∂z uᴱ = −(u'²/w'²) ∂z uˢ; with isotropic turbulence this homogenizes uᴸ = uᴱ + uˢ. Measure
   how far a single group gets.
3. **Experiments 2 and 3 (regular waves).** Steady wave trains over turbulence, longer fetch:
   does the response approach quasi-equilibrium?
4. **Numerical hygiene.** Packet-only nulls, matched controls, timestep and resolution
   convergence, closure sensitivity, and a bounded-tank variant with the physically correct wall
   condition uᴸ·n = 0.

The paper's tabulated values live in `cases.jl` (Tables 2 and 3, transcribed from the table
images at Cambridge Core; the text-summarizing fetch tools hallucinated numbers, so always read
the table PNGs `S002211202611163X_tab{1..5}.png`).

## 2. Physical and numerical design

* **Frame.** The Galilean frame following the unperturbed current. No mean flow, no wind stress,
  no buoyancy, no rotation, no Stokes-streaming forcing. Turbulence decays in time as it decays
  downstream in the flume; the field of view (FOV) corresponds to a time after the grid.
* **Waves.** Prescribed through `Oceananigans.StokesDrift` (packet) or `UniformStokesDrift`
  (regular waves). Oceananigans prognoses the Lagrangian-mean velocity uᴸ; the Eulerian mean is
  uᴱ = uᴸ − uˢ.
* **Packet** (`moving_packet.jl`): long-crested Gaussian Stokes envelope G(ξ) = exp(−ξ²/σ₀²),
  ξ = x − x₀ − c_g t, with the finite-depth solenoidal completion
  wˢ = −(Uˢ₀/2k) G′(ξ) [e^{2kz} − e^{−2kh}] (zero at the bottom). Periodic tank: three-image sum;
  bounded tank: single Gaussian. Derivatives are unit-tested against finite differences and the
  discrete divergence converges at second order.
* **Tank.** 12 m × 0.8 m × 0.4 m for Experiment 1 (12 m so that the packet starts 4σ₀ upstream
  of the plane at x = 6 m and stops 4σ₀ downstream without re-entering). Regular waves: doubly
  periodic 3.2 m × 3.2 m × h box.
* **Grid.** `RectilinearGrid`, stretched vertically (refinement 1.5, stretching 8) with the
  finest cells at the surface. Levels in `common.jl`: S0 384×32×48, M0 768×64×64, M1 1024×96×96,
  M2 1536×128×128, M3 2048×192×160; boxes RT/R0/R1/R2. Float32.
* **Numerics.** WENO(order=5) with no explicit closure (implicit LES), RK3, fixed Δt = 0.02 s.
  Sensitivities: `numerics=amd` (Centered + AnisotropicMinimumDissipation), `weno_nu`, `weno9`.
* **Members.** `quiescent_control`, `packet_null` (or `waves_null`), `turbulence_control`,
  `packet_turbulence` (or `waves_turbulence`). Pairs share the checkpoint, grid, Δt and an
  iteration-based output schedule, so differences are exact in time.
* **Residual.** ΔU = (U_packet+turb − U_turb) − (U_packet+null − U_null). The null removes the
  irrotational return flow and any numerical response to the prescribed forcing.
* **Initialization.** uᴸ₀ = uᴱ_turb + uˢ₀, volume mean removed, then `set!` (pressure projection).
  In a periodic tank the volume-mean removal imposes the zero net Lagrangian transport that a
  closed tank imposes physically; the resulting return flow is localized under the packet and
  equals the paper's Table 3 `u_rf` for regular waves.
* **Turbulence** (`generate_turbulence.jl`): random solenoidal Fourier field with a von Kármán
  spectrum cut off at 6 cells, energy-containing wavenumber calibrated so that the generated
  L₁₁ = L_factor × L_case, per-component rms enforced after the pressure projection (three
  rescale/re-project rounds), w tapered at top and bottom, 2 s of preconditioning, saved as a
  checkpoint. Per-case amplitude multipliers are calibrated so the no-wave control matches the
  case rms at the FOV time (see README table). In the 0.8 m × 0.4 m cross-section the streamwise
  integral scale saturates near 0.23 m whatever the target, because divergence-free modes that
  are long in x must carry their energy in v and w.

## 3. Where things are

```
experiments/anti_stokes/
  cases.jl                     measured values and derived parameters for 1.A–1.D and 2.*, 3.*
  moving_packet.jl             packet envelope, solenoidal completion, StokesDrift derivatives,
                               analysis_windows, snapshot_times
  common.jl                    levels, grid (x_topology periodic|bounded), model, numerics,
                               CLI parsing, directory layout, metadata, SpecifiedIterations
  generate_turbulence.jl       checkpoint generator (+ CLI)
  moving_packet_experiment.jl  run_member(...) for Experiment 1 members
  run_moving_packet.jl         CLI wrapper
  regular_waves.jl             run_regular_member(...) for Experiments 2/3 (+ CLI)
  stage_S0.jl                  all S0 members in one session + acceptance report; also used
                               for calibration (skip=1,2,3,4,7)
  smoke_test_moving_packet.jl  CPU plumbing test (T0 level)
analysis/anti_stokes/
  common.jl                    load_run, Eulerian subtraction, central moments, windows,
                               wake_age_composite, paired_residual
  quick_checks.jl              Makie-free acceptance reports (null, convergence, turbulence,
                               pair) and ensemble()
  ensemble_profile.jl          per-level ensembles (levels 1–4 on one figure)
  compare_packet_control.jl    the decisive 8-panel paired figure
  packet_hovmoller.jl          trajectory + Hovmöller diagrams (fixed and packet coordinates)
  turbulence_statistics.jl     rms, anisotropy, decay, integral scale, spectrum, u'w'
  momentum_budget.jl           wake-age composite budget ∂tΔU vs −∂zΔ⟨u'w'⟩
  lagrangian_mean.jl           uˢ, ΔU and uˢ+ΔU beneath the packet; shear ratios R, A
  compare_cases.jl             cross-case figure (absolute, normalised, vs TKE, spin-up)
  animate_packet_turbulence.jl MP4 from the seed-1 slices
  run_stage_analysis.jl        one-session driver: reports + all figures (+ animation)
  regular_common.jl            Experiments 2/3: profiles, ΔU, ensembles, turbulence report
  regular_waves_analysis.jl    Experiments 2/3 family figures
batch/
  env.sh                       PATH (julia 1.11.9 in ~/opt), unset LD_LIBRARY_PATH, threads
  precompile.batch             instantiate + precompile on a GPU node
  anti_stokes_smoke.batch      stage_S0.jl with pass-through args (also calibration)
  anti_stokes_generate_ic.batch  checkpoint array (seeds), env CASE LEVEL AMPLITUDE L_FACTOR X_TOPOLOGY
  anti_stokes_marginal.batch   Experiment 1 array: 0 null, 1 null Δt/2, 2 quiescent, 11–1n seeds,
                               21 seed 1 at Δt/2; env CASE LEVEL NUMERICS X_TOPOLOGY AMPLITUDE
  anti_stokes_regular.batch    Experiments 2/3 array: 0 null, 2 quiescent, 11–14 seeds, 90 calibration
  anti_stokes_analysis.batch   post-processing on a compute node (stage driver or script=<name>)
test/                          2190+ unit tests (julia --project=. test/runtests.jl)
figures/anti_stokes/           committed PNGs, compact MP4s, gallery.html (GitHub Pages)
logs/                          SLURM logs (gitignored)
```

Data root: `/work/hdd/bhcr/glwagner/anti_stokes/case_<name>/<level>/<member>/<tag>/` with tag
`seed_0001_dt0.020[_amd][_LxX_LyY][_boundedx]` (or `quiescent_...`). Each run directory holds
`metadata.jld2` (everything needed to reproduce it, including the checkpoint sha256),
`y_averages.jld2` (Experiment 1: ⟨u⟩,⟨v⟩,⟨w⟩,⟨u²⟩,⟨v²⟩,⟨w²⟩,⟨uw⟩ over y every 0.1 s) or
`profiles.jld2` (Experiments 2/3: horizontal averages), `fov_plane.jld2` (y–z plane at the FOV),
`statistics.jld2` (scalar time series: uˢ at the FOV, packet centre, volume mean/rms, max |w|),
`snapshots.jld2` (3D u,v,w at t_peak + {−3,−1,0,1,3,4}τ₀), `run_summary.jld2`, and for seed 1
`xz_slice.jld2` and `xy_surface.jld2` for animations. Checkpoints:
`case_<family>/<level>/initial_conditions/seed_XXXX[_tags].jld2` (u, v, w interior arrays +
metadata with amplitudes, k_e, rms before/after projection and preconditioning).

## 4. How the existing data were generated (chronological, all on DeltaAI GH200 nodes)

1. `sbatch batch/precompile.batch` (GLMakie fails headless; harmless).
2. **S0** (384×32×48): `sbatch batch/anti_stokes_smoke.batch` → quiescent, null at Δt and Δt/2,
   null at doubled x-resolution (S0x), one crude pair. All null criteria met.
3. **Calibration** of the 1.D turbulence at M0 by repeated
   `sbatch batch/anti_stokes_smoke.batch level=M0 amplitude=a,b,c skip=1,2,3,4,7` (generates
   seed 1 with overwrite and runs the control; read `amplitude multipliers ... → suggested` in the
   turbulence report). Frozen: `1.24,0.80,1.20` (L_factor 1.15).
4. **M0/M1/M2 for 1.D**:
   `IC=$(AMPLITUDE=1.24,0.80,1.20 LEVEL=M0 sbatch --parsable batch/anti_stokes_generate_ic.batch)` then
   `AMPLITUDE=... LEVEL=M0 sbatch --dependency=afterok:$IC batch/anti_stokes_marginal.batch`
   (M1 with `--array=0-2,11-14` and `NUMERICS=amd --array=0,11-12`; M2 with `--array=1-8` for the
   checkpoints and `--array=0-2,11-18`).
5. **Cases 1.A, 1.B, 1.C.1, 1.C.2** at M2: per-case calibration (2–3 rounds each,
   `case=1.A level=M2 amplitude=... L_factor=... skip=1,2,3,4,7`), then checkpoints and
   `--array=0-2,11-14`. Frozen amplitudes in the README table. 1.C.1 and 1.C.2 use identical
   checkpoints (same amplitudes and seeds).
6. **Bounded tank** for 1.D at M2: `X_TOPOLOGY=bounded`, calibration in four rounds (the 23 s
   approach decays the turbulence more; frozen `2.85,0.99,2.90`, L_factor 1.6, 8–15 % low), nulls
   `--array=0-2`, pairs `--array=11-14`.
7. **Analysis** on compute nodes: `sbatch batch/anti_stokes_analysis.batch case=<c> level=M2
   seeds=1,2,3,4 [x_topology=bounded] [animation=true]`, plus `script=compare_cases.jl ...`,
   `script=lagrangian_mean.jl ...`.
8. **Experiments 2/3** (in progress when this was written): see §6.

GPU cost so far: about 10 GPU-hours. A member costs 1.5 min (S0) to 2.4 min (M2), 6 min for the
46 s bounded runs; compilation adds ~1.5 min per Julia process.

## 5. Results to date (for context; the PR description has the full tables)

* 1.D composite surface residual: −10.9 ± 0.8 (M0), −10.5 ± 0.3 (M1), −11.5 ± 0.4 mm/s (M2,
  eight seeds); AMD closure −12.05; bounded tank −10.1 ± 0.4. Timestep halving changes it by
  0.001 mm/s. Zero crossing k₀z ≈ −0.75, compensating deep lobe, ∫ΔU dz ≈ 0.
* Budget closes: ∂tΔU ≈ −∂zΔ⟨u'w'⟩ within a few percent in wake-age composites.
* 1.A −4.39 ± 0.03, 1.B −6.93 ± 0.12, 1.C.1 −8.34 ± 0.17, 1.C.2 −13.72 ± 0.31 mm/s; the jet
  depth orders with the integral scale; steepness ratio 1.65 ± 0.05 vs 2.15 for ϵ².
* Lagrangian mean is not homogenized within one group: 7 % of the Stokes shear is cancelled at
  the packet centre, ~50 % by +0.9 τ₀; the current keeps growing 1–2 τ₀ after the packet.
* Regular waves (Experiments 2/3, R1, four seeds, onset-matched turbulence): at t_FOV the surface
  Eulerian change is 0.67–0.92 of Uˢ₀ for 2.A, 0.53–0.77 for 2.B, 0.52–0.75 for 3.A and
  0.41–0.67 for 3.B, with shear ratios R = 0.5–1.1 through the Stokes layer, i.e. the Lagrangian
  mean is close to homogenized after ~30 s of steady waves over energetic turbulence and still
  rising at t_FOV; full quasi-equilibrium (R = A ≈ 3–6) is not reached. ΔU/Uˢ₀ decreases with
  steepness in every family. The wave-only null reproduces Table 3's return flow u_rf.

## 6. Reproducing and extending

### Environment

```bash
export PATH=/u/glwagner/opt/julia-1.11.9/bin:$PATH        # or source batch/env.sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'       # on a compute node, not the login node
julia --project=. test/runtests.jl                        # ~1 min after precompilation
```

SLURM account `bhcr-dtai-gh`, partition `ghx4` (GH200, 1 GPU per member). Never run heavy
Julia on the login node (38 GB / 1000-task cgroup shared with your own session); use
`batch/anti_stokes_analysis.batch` or `sbatch --wrap`. Never `pkill -f <pattern>` from a shell
whose own command line contains the pattern.

### A new wave-group case

1. Add a `wave_group_case(...)` line in `cases.jl` and register the name in
   `ANTI_STOKES_CASES`; add its test values to `test/test_moving_packet.jl`.
2. Calibrate turbulence at the target level: run
   `sbatch batch/anti_stokes_smoke.batch case=X level=M2 amplitude=a,b,c L_factor=f skip=1,2,3,4,7`,
   read the report's suggested multipliers from the log, repeat until the rms ratios at t_peak
   are within 10 %. Expect 2–4 rounds; energetic turbulence decays faster, so returns diminish.
3. Checkpoints and pairs: `CASE=X LEVEL=M2 AMPLITUDE=a,b,c L_FACTOR=f sbatch --array=1-4
   batch/anti_stokes_generate_ic.batch`, then the marginal array with `--dependency=afterok`.
4. Analysis: `sbatch batch/anti_stokes_analysis.batch case=X level=M2 seeds=1,2,3,4 animation=true`,
   then `script=compare_cases.jl level=M2 cases=...`.

### A new resolution level or numerics

Add to `RESOLUTION_LEVELS` in `common.jl` (halo must be ≤ the smallest dimension; WENO9 needs
halo 5) or to `numerics_settings`. Every member of a comparison must share the level and Δt.
Nulls are per level and per numerics, not per seed.

### Bounded tank

Add `X_TOPOLOGY=bounded` to the environment of the batch scripts (or `x_topology=bounded` on the
CLI). Trajectory runs from −4σ₀ to Lx + 4σ₀ (46 s for 1.D), t_peak = (x_FOV + 4σ₀)/c_g, windows
are relative to t_peak, checkpoints carry the `_boundedx` tag and Nx + 1 u-faces.

### Regular waves (Experiments 2/3)

```bash
# calibrate a family's turbulence at R1 (checkpoint with overwrite + control + report)
CASE=2.A.1.3 LEVEL=R1 AMPLITUDE=1.6,1.4,1.5 L_FACTOR=1.2 sbatch --array=90 batch/anti_stokes_regular.batch
# production for one case (its family checkpoints are shared by all its steepnesses)
CASE=2.A.1.3 LEVEL=R1 AMPLITUDE=... sbatch batch/anti_stokes_regular.batch
# figures for a family
sbatch batch/anti_stokes_analysis.batch script=regular_waves_analysis.jl family=2.A level=R1 seeds=1,2,3,4
```

Open assumptions to revisit: v_rms = u_rms (planar PIV), L = 0.15 m for Experiment 2, FOV time
t_FOV = 8.5 m / U₀ (28.3 s for Experiment 2, 44.7 s for Experiment 3), Lagrangian initialization
with waves present from t = 0.

**Turbulence protocol for Experiments 2/3 (important).** Freely decaying box turbulence cannot
reproduce the measured rms after 28–45 s of decay: two calibration rounds showed the rms at
t_FOV is decay-saturated (doubling the initial amplitude left it unchanged at ~0.6 of the 2.A
target and ~0.4 of the 3.A target, while L₁₁ grew to 2–3 times the measured value). The flume
values (u_rms ≈ 25 mm/s at U₀ = 0.30 m/s in 0.80 m; ≈ 12 mm/s at 0.19 m/s in 0.50 m) are
consistent with u_rms ≈ 2u* of the bottom boundary layer, i.e. turbulence maintained by the
channel walls rather than decaying grid turbulence. The first-pass protocol therefore matches the
measured rms and L₁₁ at wave onset (amplitude ≈ 1.1–1.2, L_factor 1.0), integrates for the
laboratory interaction time, and reports the response as a function of time since onset
together with the decaying turbulence intensity (`regular_report` prints both at 0.1–1.0 t_FOV).
The faithful fix is statistically stationary turbulence, implemented as the second protocol.

**Stationary (band-forced) turbulence protocol, `forcing=band`.** `forced_turbulence.jl` forces
the horizontal velocity fluctuations at large horizontal scales, F_u = γ_u ℬ[u], F_v = γ_v ℬ[v],
where ℬ is a horizontal band-pass filter over k_h ∈ [0.5, 1.5] k_e with k_e = 0.75/L (batched 2D
FFTs, applied at every level; the horizontal mean k_h = 0 is never forced, w is not forced). A
checkpoint is produced by a closed-loop spin-up of 8 eddy turnovers (L/u_rms) in which the gains
hold the volume rms of u and v at the case values with a proportional–integral law on the
relative energy error e = (target² − rms²)/target²: γ_i = max(0, γ_ref (1 + 3e) + I_i),
dI_i/dt = κ e, with γ_ref = u_rms/L and κ = γ_ref per eddy turnover (integral frozen while the
gain is clamped at zero). The gains are then frozen at their average over the last quarter of the
spin-up (`γ_open` in the checkpoint metadata) and the experiment members run in open loop, so the
forcing is the same function of the flow in the wave and control members and cannot mask a
wave-induced change of the turbulence. Forced members and checkpoints carry the `_forced` tag
(`extra="forced"` in `run_directory`/`initial_condition_path`; `extra=forced` in the analysis CLI).

```bash
# spin up the family checkpoint (seed 1) and run the forced control with the report
CASE=2.A.1.3 LEVEL=R1 FORCING=band sbatch --array=90 batch/anti_stokes_regular.batch
# production with stationary turbulence (checkpoints per seed are spun up on demand under flock)
CASE=2.A.1.3 LEVEL=R1 FORCING=band sbatch batch/anti_stokes_regular.batch
sbatch batch/anti_stokes_analysis.batch script=regular_waves_analysis.jl family=2.A level=R1 seeds=1,2,3,4 extra=forced
# plumbing test (unit tests + RT-level spin-up, members and report on a GPU, scratch data root)
sbatch batch/anti_stokes_forced_test.batch
```

### Plotting recipes

All scripts take `key=value` arguments and write PNGs to `figures/anti_stokes/` with the case,
level and topology in the name. Figures and MP4s are gitignored; commit the ones you want with
`git add -f`. The gallery page `figures/anti_stokes/gallery.html` is served by GitHub Pages from
the branch and loads the compact MP4s through jsDelivr (12 h cache).

* Decisive paired figure: `compare_packet_control.jl packet=<dir> control=<dir> null=<dir> quiescent=<dir>`
* Ensembles across levels: `ensemble_profile.jl case=1.D level=M0 seeds=1,2,3,4 level2=M1 level3=M2 seeds3=1,...,8`
* Cross-case: `compare_cases.jl level=M2 cases=1.A,1.B,1.C.1,1.C.2,1.D seeds=1,2,3,4 seeds_1.D=1,...,8`
* Lagrangian mean: `lagrangian_mean.jl case=1.D level=M2 seeds=1,...,8 [x_topology=bounded]`
* Budget: `momentum_budget.jl packet=... control=... null=... quiescent=...`
* Animation (needs seed-1 slices, i.e. `animation=true` at run time): `animate_packet_turbulence.jl packet=<dir> control=<dir>`;
  compress with Makie's bundled ffmpeg (`CairoMakie.Makie.FFMPEG_jll.ffmpeg()`), e.g.
  `-vf scale=1280:-2 -c:v libx264 -crf 30`.
* Everything at once for a stage: `run_stage_analysis.jl case=<c> level=<l> seeds=<s> [x_topology=bounded] [animation=true]`.

### Statistics: use the wake-age composite

The paired difference at a single point is dominated by turbulence decorrelation (the packet's
return flow displaces the turbulence relative to the control). `wake_age_composite` averages all
(x, t) with the same age (x_c(t) − x)/c_g, i.e. ~60 integral scales per age bin, and is the
low-noise estimate; report it alongside the laboratory-style FOV before/after window. Ages
beyond the run's reach are NaN.

### Acceptance criteria (from the campaign document; all met so far)

Packet null: trajectory error ≤ 1e-6, post-packet FOV residual < 0.1 mm/s, return-flow transport
ratio −1.00, Δt-halving change < 5 %, no far-field artefact. Turbulence: rms within 10–15 % (M0)
or 5–10 % (M2) of the case, L₁₁ within ~20 % where the tank allows. Signal: negative near the
surface, k₀|z| = O(1), small depth integral, ensemble mean ≫ standard error, robust to Δt,
resolution and closure.

## 7. Pitfalls recorded during the campaign

* `using CUDA` is required in `common.jl` for `GPU()`; JLD2 needs `iotype=IOStream` on Lustre
  (`JLD2_KW`); use `sum/length` rather than `Statistics.mean` on GPU views.
* Iteration-based schedules (`IterationInterval`, `SpecifiedIterations`) keep paired members
  exact; `TimeInterval` would perturb Δt when the interval is not a multiple of Δt.
* Plain `mean` over cells over-weights the refined surface cells; use `volume_mean` (an
  `Average` field) for transports.
* In a session that `include`s the figure scripts (`run_stage_analysis.jl`) a script edited after
  the session started sees stale definitions; rerun in a fresh process.
* Julia buffers stdout when redirected; reports appear only when the process ends. SLURM logs
  likewise. Check `sstat`/`ps` rather than the log to see whether a job is alive.
* The M2 analysis of a 46 s bounded run reaches ~50 GB resident: compute node only.
* Two Claude instances on different login nodes wrote the same files at once; keep one instance
  (tmux on gh-login01) and kill orphans (`for n in gh-login0{1..4}; do ssh $n pgrep -fa claude; done`).
