# Research objectives: velocity profiles driven by wind stress beneath surface waves

## The question

When wind blows over the ocean, it drives a near-surface shear flow. Surface waves modify this flow through the Craik-Leibovich (CL) vortex force, which drives Langmuir circulations and alters the mean velocity profile U(z). The central question is: **what is the shape of U(z) in the presence of surface waves, and can it be described by a modified similarity theory?**

In the classical (no-wave) boundary layer, the velocity profile follows Monin-Obukhov similarity:

    dU/dz = (u* / κz) φ_m(z/L)

where φ_m = 1 in neutral conditions (log law). Waves change this picture in at least two ways:

1. **Langmuir turbulence** — the CL vortex force drives overturning cells that enhance vertical mixing, effectively reducing the apparent friction velocity u*_eff and increasing the apparent roughness length z₀_eff (Teixeira 2018).

2. **Stokes drift shear production** — Stokes drift shear produces TKE alongside the mean shear, so the nondimensional shear φ_m should depend on a wave parameter (e.g., turbulent Langmuir number Laₜ) in addition to stability z/L. Large et al. (2019) proposed a separable form: φ_m = φ(ζ) × χₛ(ξ), where ξ measures the ratio of Stokes shear TKE production to total production.

These theories have been tested in large-eddy simulations of ocean-scale boundary layers (Sullivan, McWilliams, Patton, Large, Romero, etc.) but **not at laboratory scale**, where the wave parameters (wavelength, steepness, Laₜ) are very different. Wagner et al. (2023) simulated the lab-scale transition to Langmuir turbulence but focused on the instability onset, not the quasi-steady velocity profiles.

## What we want to do

Run simulations closely matching W23 (same domain, wave parameters, wind forcing, CL equations via Oceananigans/NumericalWaveTanks.jl) and **systematically extract and analyze the mean velocity profile U(z)** as it evolves from the laminar wind-drift solution through transition to a quasi-steady turbulent state.

### Primary diagnostics

For each simulation, output time series of horizontally-averaged profiles:

- **U(z, t)** — mean streamwise velocity (the main target)
- **V(z, t)** — mean cross-stream velocity
- **⟨u′w′⟩(z, t)** — turbulent momentum flux (Reynolds stress)
- **TKE(z, t)** — turbulent kinetic energy
- **ε(z, t)** — dissipation rate

From the quasi-steady profiles, compute:

- **Nondimensional shear:** φ_m(z) = (κz / u*) dU/dz
- **Effective friction velocity** u*_eff and **roughness length** z₀_eff from log-layer fits
- **Stokes production ratio** ξ = ⟨u′w′⟩ ∂uₛ/∂z / (⟨u′w′⟩ ∂U/∂z + ⟨u′w′⟩ ∂uₛ/∂z)
- **Langmuir number** Laₜ = √(u* / Uₛ₀)

### Questions to answer

1. Does the quasi-steady velocity profile follow a log law? If so, what are u*_eff and z₀_eff, and do they depend on Laₜ as Teixeira (2018) predicts?

2. Does the Large et al. (2019) separable similarity form φ_m = φ(ζ) × χₛ(ξ) hold at laboratory scale, where Laₜ is O(1) rather than O(0.3) as in the ocean?

3. How does near-surface dissipation scale with depth — does Craig & Banner's ε ~ z^{-1.75} hold, or does the lab-scale wave field produce a different scaling?

4. Is the Stokes drift correction to the velocity profile primarily a surface effect (confined to depths ~ 1/2k), or does it penetrate deeper through the turbulent momentum flux?

## Stratification

The same simulations are extended to include a two-layer density structure (fresh upper layer, salty lower layer) with a tanh thermocline. This adds a buoyancy dimension to the problem.

### Additional diagnostics for stratified runs

- **b(z, t)** — mean buoyancy profile
- **⟨w′b′⟩(z, t)** — turbulent buoyancy flux
- **Ri_g(z, t)** — gradient Richardson number = N² / (dU/dz)²
- **Mixed layer depth** h(t) — diagnosed from the buoyancy profile

### Additional questions

5. Does stratification modify the velocity profile shape in the mixed layer, or does it simply limit the depth over which the log-layer form applies?

6. Does the Businger-Dyer stability correction φ_m = 1 + βz/L apply on the water side of the air-sea interface, and if so, what is β? (This is poorly constrained observationally.)

7. Does stratification suppress or modify Langmuir circulations, and if so, how does this feed back onto U(z)?

8. Can the nondimensional shear collapse onto a single curve when parameterized by both Laₜ (wave forcing) and z/L (stability)?

## Connection to remote sensing

These simulations produce "truth" velocity profiles U(z) at high resolution. A downstream application (not part of the simulation work, but motivating it) is to test whether Doppler-based remote sensing methods — specifically the Polynomial Effective Depth Method (PEDM; Smeltzer et al. 2019) and Ensemble Kalman Inversion (EKI) — can recover U(z) from synthetic Doppler shift observations. The forward model for this is:

    c̃(k) = ∫ U(z) Q(z, k) dz

where Q(z, k) = 2k exp(2kz) is the depth-weighting kernel (Stewart & Joy 1974). Each wavenumber k samples the current at an effective depth z_eff = -1/(2k). By computing c̃(k) from the simulated U(z), we can generate synthetic observations and test the inversion methods.

This is a separate analysis step, not part of the simulation code — but the simulation diagnostics (especially the high-resolution U(z) profiles) are designed with this application in mind.

## Simulation approach

- **Code:** Oceananigans.jl via NumericalWaveTanks.jl experiment scripts
- **Physics:** CL wave-averaged equations with prescribed Stokes drift (monochromatic deep-water capillary-gravity waves)
- **Numerics:** WENO(order=9) advection (implicit LES), no explicit subgrid closure, molecular viscosity/diffusivity only
- **Initial conditions:** Laminar Ekman-Stokes wind-drift profile at t₀ = 16 s (from W23), plus small random noise perturbations
- **Resolution:** Moderate grid (~128³ to 256³), stretched near the surface
- **Platform:** GPU (NVIDIA GH200 on DeltaAI, NCSA)

See [`claude_code_plan.md`](claude_code_plan.md) for the full implementation plan and experiment matrix.
