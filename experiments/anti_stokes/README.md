# Moving-wave-packet anti-Stokes campaign

For the full record of how the data were generated, what the goals are, and how to reproduce
or extend the campaign (written for agents), see [`CAMPAIGN.md`](CAMPAIGN.md).

Numerical reproduction of the wave-group experiments of Ellingsen, Rømcke, Smeltzer, Teixeira,
van den Bremer, Moen and Hearst, *Turbulence-induced anti-Stokes flow: experiments and theory*,
J. Fluid Mech. 1029, A6 (2026), starting with Experiment 1, case 1.D.

A prescribed, solenoidal, long-crested Gaussian Stokes-drift packet moves at its group velocity
through decaying turbulence in a 12 m × 0.8 m × 0.4 m periodic tank (current-following frame),
crossing a fixed virtual observation plane at x = 6 m. The Craik–Leibovich model has no wind
stress, buoyancy, Coriolis force or Stokes-streaming forcing; the packet enters only through
`Oceananigans.StokesDrift`.

## Members

| member | turbulence | packet | purpose |
|---|---|---|---|
| `quiescent_control` | no | no | regression test (must stay at rest) |
| `packet_null` | no | yes | irrotational return flow, initialization and numerical response |
| `turbulence_control` | checkpoint | no | reference turbulence |
| `packet_turbulence` | same checkpoint | yes | the experiment |

The turbulence-induced residual is
`ΔU = (U_packet+turb − U_turb) − (U_packet+null − U_null)`.

## Files

| file | content |
|---|---|
| `cases.jl` | measured case values and the derived packet parameters |
| `moving_packet.jl` | periodic Gaussian packet, solenoidal `wˢ`, derivatives for `StokesDrift` |
| `common.jl` | resolution ladder (S0–M3), grid, model, numerics options, CLI parsing, directory layout, metadata |
| `generate_turbulence.jl` | projected von Kármán initial turbulence, integral-scale calibration, per-component rms, preconditioning, checkpoint |
| `moving_packet_experiment.jl` | `run_member`: initialization, output writers (y-averaged moments, virtual PIV plane, 3D snapshots, animation slices), metadata |
| `run_moving_packet.jl` | command-line entry point |
| `stage_S0.jl` | all S0 members in one session plus acceptance report |
| `smoke_test_moving_packet.jl` | CPU plumbing test |

## Running

```bash
# unit tests
julia --project=. test/runtests.jl

# CPU plumbing test
julia --project=. experiments/anti_stokes/smoke_test_moving_packet.jl

# stage S0 on one GPU (DeltaAI)
sbatch batch/anti_stokes_smoke.batch

# one member
julia --project=. experiments/anti_stokes/run_moving_packet.jl case=1.D member=packet_null level=M0
julia --project=. experiments/anti_stokes/generate_turbulence.jl level=M0 seed=1 amplitude=1.24,0.80,1.20
julia --project=. experiments/anti_stokes/run_moving_packet.jl member=packet_turbulence level=M0 seed=1 animation=true

# stage M0/M1/M2 arrays
IC=$(AMPLITUDE=1.24,0.80,1.20 LEVEL=M0 sbatch --parsable batch/anti_stokes_generate_ic.batch)
AMPLITUDE=1.24,0.80,1.20 LEVEL=M0 sbatch --dependency=afterok:$IC batch/anti_stokes_marginal.batch

# analysis of a completed stage (reports, ensemble, figures, animation)
julia --project=. analysis/anti_stokes/run_stage_analysis.jl level=M0 seeds=1,2,3,4
```

Output goes to `<root>/case_1D/<level>/<member>/<tag>/` with `root` defaulting to
`/work/hdd/bhcr/glwagner/anti_stokes` on DeltaAI (or `data/anti_stokes` elsewhere).

## Cases and calibrated turbulence amplitudes (M2)

Measured values from Tables 2 and 3 of the paper; the amplitude multipliers scale the (u, v, w)
rms of the generated field so that the no-wave control matches the case at t_peak = 4τ₀, and
`L_factor` scales the integral scale handed to the generator.

| case | k₀ (m⁻¹) | τ₀ (s) | (u, v, w)_rms (mm/s) | L (m) | ϵ | amplitude | L_factor | control at t_peak |
|---|---|---|---|---|---|---|---|---|
| 1.A | 9.5 | 2.4 | 7.1, 6.8, 5.8 | 0.051 | 0.20 | 1.21, 1.11, 1.07 | 1.3 | rms ratios (0.99, 1.00, 1.05), L₁₁ = 0.055 m |
| 1.B | 9.2 | 2.6 | 11, 10, 7.3 | 0.26 | 0.20 | 1.12, 0.80, 1.15 | 1.15 | (1.04, 0.96, 0.97), L₁₁ = 0.23 m |
| 1.C.1 | 8.9 | 2.9 | 16, 12, 9.2 | 0.32 | 0.15 | 1.18, 0.73, 1.20 | 1.4 | (0.99, 1.02, 1.00), L₁₁ = 0.23 m |
| 1.C.2 | 9.0 | 2.9 | same as 1.C.1 | 0.32 | 0.22 | 1.18, 0.73, 1.20 | 1.4 | identical checkpoints to 1.C.1 |
| 1.D | 9.3 | 2.8 | 17, 17, 13 | 0.20 | 0.22 | 1.24, 0.80, 1.20 | 1.15 | (1.01, 1.03, 0.94), L₁₁ = 0.20 m |

The streamwise integral scale saturates near 0.23 m in the 0.8 m × 0.4 m cross-section whatever
the generator target, because divergence-free modes that are long in x must carry their energy in
v and w; cases 1.B and 1.C therefore run with L₁₁ 12–30 % below the measured value while matching
the rms components. Cases 1.C.1 and 1.C.2 share the same checkpoints (same seeds and amplitudes).

## Notes

* All members of a pair share the grid, fixed timestep, numerics and iteration-based output
  schedule, so differencing is exact in time.
* In a periodic tank the packet's mean Stokes transport would otherwise be balanced by a uniform
  Eulerian current `Q/h ≈ 1.4 mm/s` everywhere. `remove_mean_transport=true` removes the
  volume-mean velocity at initialization, so the return flow is localized under the packet as in
  a long laboratory tank.
* The turbulence amplitude multipliers `(1.24, 0.80, 1.20)` for (u, v, w) were calibrated at M0 so
  that the no-wave control matches the case-1.D rms velocities at t = t_peak within 5 %, with the
  rms enforced after the pressure projection (the narrow tank redistributes energy between
  components). The integral scale is calibrated to 0.20 m (`L_factor` compensates for its
  decrease during decay).
* The paired difference at a single point is dominated by turbulence decorrelation, because the
  packet's return flow displaces the turbulence relative to the control. The wake-age composite
  (`analysis/anti_stokes/common.jl: wake_age_composite`) averages all `(x, t)` with the same time
  since passage and is the low-noise estimate; the FOV before/after profile reproduces the
  laboratory measurement.
