# Vallis narrow-ring decay in two-dimensional WC25 wave-averaged flow — report

Notation: u′_rms(t) = √⟨|𝐮′_L(t)|²⟩ is the total rms velocity fluctuation; u′_rms,0 is its initial value. q is reserved for potential vorticity. Historical code/configuration keys are unchanged.

*Status: COMPLETE (2026-10-03 04:30 UTC). Figures: `figures/wc25_vallis2d/final_*.png` (on disk in the worktree; PNGs are gitignored), metrics `final_metrics.txt`. Total cost ≈ 2.1 billed GPU-hours of the 8 allotted. Agent: Claude (Fable 5.1), tmux `wave-agents:vallis-2d`, DeltaAI.*

## 1. Question
Does the two-dimensional (x–z, y-invariant) wave-averaged dynamics of Wagner & Constantinou (2025, "WC25"),
started from the narrow spectral ring of Vallis & Maltrud (1993) / Vallis (2017, Figs 11.8, 12.3, 12.4),
reproduce the β-plane phenomenology (anisotropic "dumbbell" spectrum, zonal filaments and jets) for the
shallow (constant-curvature) and deep Stokes profiles, with the wave-free control as the 2-D-turbulence reference?

## 2. Setup
- **Equations.** WC25 wave-averaged Navier–Stokes for the Lagrangian-mean velocity uᴸ, unit square, periodic in x,
  impermeable free-slip walls at z = 0, 1, WENO(9) implicit LES, RK3, no closure/rotation/stratification/stress/forcing.
  With ω_y = ∂z u − ∂x w the curl of the momentum equation gives **D(ω_y − ∂z uˢ)/Dt = 0**, i.e. Dω_y/Dt = (∂zz uˢ) w:
  a β-plane with z ↔ y, w ↔ v, ζ = −ω_y and **β_eff(z) = ∂zz uˢ**. The Stokes momentum terms ∂z uˢ (w x̂ − u ẑ) do no work on uᴸ.
- **Families.** shallow ∂z uˢ = z/2 ⇒ β = 1/2 (exact β-plane; the walls are the odd-symmetric subspace of a doubly periodic 1 × 2 β-plane);
  deep ∂z uˢ = ¼e^{8(z−1)} ⇒ β(z) = 2e^{8(z−1)} (surface-intensified, variable β — NOT a homogeneous-β reproduction); none = control.
  All three start from the identical Lagrangian field; uˢ is not added. WC25 prints both signs of z/2; the sign only reverses the
  Rossby phase direction (−x here).
- **Initial condition.** ψ = Σ 2Re[A e^{i2πmx}] sin(nπz), Gaussian ring E(K) ∝ exp(−(K−K₀)²/2σ²), K₀ = 2π·9 cycles/box (the Fig. 11.8/12.4
  ring; the demo code's amplitude 1/[K²+(K²−3200)²] peaks at 9.0 cycles), σ = one x-mode spacing (VM93's Z/E = 2K² gives K ≈ 12.04 for
  their ring at 12, i.e. ≈ 1 mode wide), seeded Gaussian coefficients, k_x = 0 jet modes removed (they would carry 6.0% of the energy
  and raise the anisotropy index from 0.004 to 0.057), exactly solenoidal C-grid construction (divergence 7e−18, w = ω_y = 0 on the walls).
  u′_rms,0 = sqrt(⟨u²+w²⟩) is set from rb = β/(u′_rms,0K₀²) = 1 (demo code ≈ 0.96 once its unnormalised FFT is undone; VM93 ≈ 1.96):
  u′_rms,0 = 1.56e−4, T_e = 1/(u′_rms,0K₀) = 113 time units = the Rossby time K₀/β. Physical K = 2πn/L with L = 1.
- **Validation (job 3297593).** Linear Rossby mode ω/ω_theory = 1 − 2.3e−3 with −x phase propagation; quiescent null exactly 0; momentum
  conserved to 1e−21; Stokes work 6e−17 (shallow); time stepping converged to < 1e−4 for Δt = 0.25…4; restart bitwise identical.
  Energy drifts by +0.03% per 2 T_e independently of Δt (flux-form WENO spatial error; +0.3% per 100 T_e in the deep case from the
  discrete Stokes term). **Numerics caveat:** WENO's Z-weights use ϵ = 1e−8 while the smoothness indicators here are ~1e−9, so the
  scheme is effectively a linear 9th-order upwind scheme at u′_rms,0 ~ 1e−4; a λ = 200 rescaled member (identical physics) tests this.

## 3. Runs
| job | members | status |
|---|---|---|
| 3297610 | 256² shallow (200 T_e), deep (185 T_e, checkpointed) | shallow complete; deep resumed in 3297789 |
| 3297789 | deep recompute (env truncation), figures | complete |
| 3297996 | none 256², figures | complete |
| 3297931 | 512² shallow/deep/none, 200 T_e | complete (10 min) |
| 3298003 | doubly periodic β = 1/2 reference | complete (4 min) |
| 3298023 | rb 2/0.5, width 0.5/2, zonal, λ = 200, seed 2 | complete (13 min) |
| 3298711 | final figure pass | complete |
| 3298024 | 6-seed shallow k₀ = 12, rb = 2 (+ none ×3, deep ×1), 60 T_e | complete (14 min) |
| 3298790, 3298804 | final figure passes incl. ensemble (3298804: cosmetic regeneration) | complete |

## 4. Results

### 4.1 Shallow waves (β = 1/2): the Vallis Fig. 12.3/12.4 phenomenology is reproduced
*(figures `pilot_panels_shallow.png`, `pilot_spectra_shallow.png`, `pilot_transfer.png`, `pilot_timeseries.png`)*
- **Spectrum.** From the isotropic ring at 9 cycles the energy moves to larger scales while avoiding the Rhines
  dumbbell K² = β|cos θ|/u′_rms(t): at 40 T_e it sits at the dumbbell's poles (|k_x| ≲ 3, k_z ≈ 4–8 cycles), at 200 T_e at
  |k_x| ≤ 2, k_z ≈ ±4–5 cycles. The anisotropy index (⟨k_z²⟩−⟨k_x²⟩)/(⟨k_z²⟩+⟨k_x²⟩) rises from 0.00 to 0.50 (40 T_e) and 0.70 (200 T_e);
  the energy centroid falls from 8.97 to 5.1 cycles (k_z-centroid 5.2, k_x-centroid 2.2). This is the VM93/Vallis Fig. 12.3 picture.
- **Physical space.** ω_y evolves from the isotropic blob pattern to x-elongated streaks and bands (Fig. 12.4, middle and right);
  ψ organises into x-elongated cells of alternating sign with 4–5 bands across the depth. Energy is conserved to +0.03% (no
  drag), enstrophy decays to 39% (filamentation and implicit dissipation).
- **Transfer.** The spectral transfer T(K) is negative at the ring (K ≈ 7–9) and positive at K ≈ 4–6, so the flux
  Π(K) = −Σ_{K'≤K} T is negative below the ring: a genuine inverse energy transfer, strongest at 20 T_e and weakening by 200 T_e.
  The transfer into k_x = 0 modes is small (10⁻³ of K₀/T_e).
- **Jets.** Exactly zonal (k_x = 0) flow stays weak: K_jet/K ≈ 0.05, max|U_L|/u′_rms,0 ≈ 0.7 with many fine bands (wavelength ≈ 0.09 at
  200 T_e). With rb = 1 the Rhines wavenumber √(β/u′_rms,0) equals the ring wavenumber by construction, so the cascade is arrested
  near the injection scale and energy piles up near, not on, the k_z axis — "zonally elongated structures" rather than
  finite-amplitude zonal jets within 200 T_e. The x-mean flow is therefore a poor single indicator (the wave-free control has
  a larger x-mean fraction from its box-scale condensate, §4.3).

### 4.2 Deep waves (β(z) = 2e^{8(z−1)}): depth-localised β dynamics plus a mean Lagrangian jet
*(`pilot_panels_deep.png`, `pilot_comparison_t200.png`, `pilot_jets.png`)*
- The upper ~0.3 of the domain (where the local β/(u′_rms,0K₀²) exceeds ~0.1) behaves like the shallow case: x-elongated streaks,
  anisotropy index 0.62 at 200 T_e. Below z ≈ 0.6, β is negligible and the flow is ordinary 2-D turbulence: coherent vortices,
  roll-up and merger exactly as in Vallis Fig. 11.8 — both regimes coexist in one domain.
- A strong x-mean Lagrangian jet develops: K_jet/K grows almost linearly to 0.39 at 200 T_e (0.38 at 512²), max|U_L|/u′_rms,0 = 2.2,
  with a surface jet toward −x at z ≈ 0.9 and interior jets toward +x near z ≈ 0.65 and 0.2 (4 sign reversals, wavelength ≈ 0.2).
  The transfer into k_x = 0 modes is 10× that of the shallow case and concentrated at k_z ≈ 2–4 cycles. The surface-intensified
  PV gradient ∂zz uˢ is mixed by the eddies (partial homogenisation of q = ω_y − ∂z uˢ), which requires a mean shear ∂z U_L.
- Energy drifts by +0.6% (256²) / +0.1% (512²) over 200 T_e: the discrete Stokes term does a small amount of work because of
  staggered-grid interpolation; it vanishes with resolution and is absent in the shallow and wave-free members.
  Enstrophy decays to 42% (shallow 39%).

### 4.3 Wave-free control, doubly periodic β-plane reference and resolution
*(`pilot_comparison_t200.png`, `pilot_panels_none.png`, `pilot_betaplane.png`, `pilot_variants.png`, `pilot_timeseries.png`)*
- **Control (no waves), same field.** Classical 2-D decay exactly as Vallis Fig. 11.8: roll-up, coherent vortices, like-sign merger, a
  featureless enstrophy landscape between the vortices; the energy centroid falls from 9 to 2.6 cycles (vs 5.1 with shallow waves) and
  the enstrophy to 14% (vs 39–42% with waves): waves inhibit the transfer, both to large scales (energy) and to small scales (enstrophy).
  The anisotropy index stays ≈ 0 throughout. Its x-mean fraction (0.12) comes from the box-scale condensate, not from jets.
- **β-plane reference (doubly periodic, pseudo-spectral, β = 1/2, same ring law and u′_rms,0, ∇⁸ hyperviscosity).** The ζ fields, the
  dumbbell-shaped spectra and the zonal fraction (0.060 vs 0.050) track the WC25 shallow case with walls closely (see the Hovmöller and
  final profiles): the textbook geometry and the WC25 wall geometry give the same answer at rb = 1, so walls are not what limits the jets.
  Energy decays 1% in the reference (hyperviscosity) but is conserved to +0.03% by WENO.
- **Resolution.** 512² runs from the same coefficients give K_jet/K = 0.050 (shallow; 256²: 0.050), 0.384 (deep; 0.400), anisotropy
  0.715 (0.696), identical U_L(z) profiles up to small-scale detail, and 6% more enstrophy at 200 T_e (less implicit dissipation).
  The phenomenology is converged at 256²; the deep-case energy drift falls from +0.6% to +0.1%.

### 4.4 Sensitivity (job 3298023, 256², 100 T_e each; `final_variants.png`)
| variant (shallow unless noted) | K_jet/K | anisotropy | max|U_L|/u′_rms,0 | K/K₀ |
|---|---|---|---|---|
| base rb = 1, σ = 1, seed 1 (at 100 T_e) | ≈0.04 | ≈0.62 | ≈0.5 | 1.000 |
| rb = 2 (VM93-like, wave-dominated) | 0.017 | 0.70 | 0.32 | 1.003 |
| rb = 0.5 (more turbulent) | **0.159** | 0.44 | 0.95 | 0.997 |
| ring width σ = 0.5 | 0.027 | 0.61 | 0.41 | 1.000 |
| ring width σ = 2 | 0.040 | 0.56 | 0.58 | 1.000 |
| k_x = 0 modes kept (6% initial zonal energy) | 0.097 | 0.70 | 0.80 | 1.002 |
| λ = 200 (u′_rms,0 = 0.031, identical physics; WENO nonlinear) | 0.039 | 0.63 | 0.49 | 0.990 |
| seed 2 | 0.043 | 0.64 | 0.45 | 1.001 |
| none seed 2 / none λ = 200 | 0.113 / 0.118 | 0.01 / 0.01 | 0.65 / 0.87 | 0.993 / 0.983 |
| deep seed 2 | 0.074 | 0.45 | 0.69 | 0.999 |

- **rb controls the outcome.** The exact-zonal fraction falls monotonically with rb (0.159 → ≈0.04 → 0.017) while the spectral
  anisotropy rises (0.44 → 0.62 → 0.70): a stronger β arrests the cascade closer to the ring and keeps the energy in zonally elongated
  *waves* near the dumbbell poles rather than in finite-amplitude jets; a weaker β lets the inverse cascade proceed to the Rhines scale and
  deposit energy into k_x = 0. rb ≈ 0.5 is the regime in which Vallis' late-time zonal bands would be expected within 100–200 T_e.
- **Ring width (0.5–2 modes) matters little**; the σ = 0.5 ring is slightly less jet-forming (weaker initial nonlinearity).
- **k_x = 0 seed.** Keeping the periodic ring's jet modes adds their energy (6%) almost additively to the generated zonal fraction.
- **Numerics.** λ = 200 reproduces the base results (0.039 vs ≈0.04; 0.63 vs 0.62) with 1% energy loss instead of +0.03%: the
  effectively linear WENO regime at u′_rms,0 ~ 1e−4 does not change the phenomenology, only the implicit dissipation level.
- **Realisation.** Shallow seed 2 matches seed 1; the deep mean jet is realisation-sensitive (K_jet/K 0.074 vs ≈0.2 at 100 T_e).

### 4.5 Ensemble: VM93 Fig. 5 / Vallis Fig. 12.3 analogue (job 3298024; `final_ensemble_spectra_{shallow,none}.png`, `final_ensemble_panels_*`)
Six shallow realisations with the VM93 parameters (ring k₀ = 12, σ = 1 mode, rb = β/(u′_rms,0K₀²) = 2 ≈ VM93's 1.96), 60 T_e each
(VM93's panels are at ≈ 0, 17 and 50 turnovers), averaged in spectral space; three wave-free seeds as the isotropic reference.
- **Shallow ensemble mean.** At 20 T_e the energy has left the ring and avoids the interior of the Rhines dumbbell K² = β|cos θ|/u′_rms,
  concentrating at the dumbbell's poles (|k_x| ≲ 3, k_z ≈ 5–11 cycles) with a clear hole around the k_x axis; at 60 T_e the pattern is a
  narrow hourglass along the k_z axis — the VM93 Fig. 5c / Vallis Fig. 12.3 dumbbell. Exact-zonal energy stays small
  (K_jet/K = 0.012 ± 0.003 at 60 T_e, six seeds), as in VM93's observation that the energy within the wave regime stays small while
  the zonal modes build up slowly.
- **Wave-free ensemble mean.** Isotropic collapse toward small K (no dumbbell, no preferred direction), K_jet/K = 0.06 ± 0.03 from the
  condensate's random projection on the x-mean.
- The ensemble mean removes most of the single-realisation speckle visible in the k₀ = 9 pilots and makes the dumbbell boundary sharp,
  which is why VM93 averaged six runs; the shape, not the amplitude, is the comparison target (see §5).
## 5. Limitations
- **Not a quantitative reproduction.** VM93's domain/energy conventions are inferred (E = u′_rms,0²/2, Z = ⟨ζ²⟩, 2π box) and their ring width
  only from Z/E; Vallis' Fig. 11.8/12.4 IC is known only as "a few modes around wavenumber 9" (the demo code is a related illustration,
  not the figure's code). Our ring (Gaussian, σ = 1 mode, k₀ = 9, rb = 1) is a documented choice inside the plausible bracket; the
  comparison is qualitative (dumbbell spectrum, zonal elongation), and the k₀ = 12, rb = 2 ensemble is a secondary comparison target.
- **One realisation per case** for the k₀ = 9 pilots (plus seed 2 in the sensitivity set); Fig. 12.3 is a six-run ensemble mean.
- **Finite time.** 200 T_e covers the Vallis demo's duration (≈220 turnovers) but the shallow case has not reached finite-amplitude exact-zonal
  jets; with rb = 1 the Rhines scale coincides with the injection scale by construction, so the inverse cascade is arrested near the ring.
- **Walls.** The sine basis breaks full isotropy in mode counts (k_z spacing π vs 2π) and removes k_x = 0 modes (6% of a periodic ring's energy);
  the free-slip walls admit boundary-trapped x-mean flow (visible in the deep case at z ≈ 0.9). The β-plane reference shows the walls do not
  change the shallow-case outcome at rb = 1.
- **Deep case is not a homogeneous β-plane.** β(z) = 2e^{8(z−1)} varies by 3000× across the depth; its strong x-mean jet is a response to a
  localised PV gradient, not Rhines jets. Its +0.6% (256²) / +0.1% (512²) energy drift is a discretisation residual of the Stokes term.
- **Numerics.** WENO(9) at u′_rms,0 ~ 1e−4 is effectively a linear 9th-order upwind scheme (ϵ = 1e−8 ≫ smoothness indicators); energy is
  conserved to +0.03% while enstrophy is dissipated implicitly. The λ = 200 rescaled member (v04) tests this directly. The pseudo-spectral
  reference uses ∇⁸ hyperviscosity and 2/3 dealiasing (1% energy loss), so the two codes' dissipation differs by construction.
- **Sign convention.** WC25 prints ∂z uˢ = ±z/2 in different places; we use +z/2, which fixes the Rossby phase direction (−x) but nothing else.
