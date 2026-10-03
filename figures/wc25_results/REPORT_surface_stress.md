# Surface stress on WC25 wave-averaged decaying turbulence — campaign report (2026-10-03 03:55 UTC)

Notation: u′_rms(t) = √⟨|𝐮′_L(t)|²⟩ is the total rms velocity fluctuation; u′_rms,0 is its initial value. q is reserved for potential vorticity. Historical code/configuration keys are unchanged.

**Question.** Does a constant along-wave surface stress turn the depth-alternating jets / cross-wave vortices of
initially Lagrangian-shear-free wave-averaged turbulence (Wagner & Constantinou 2025, "WC25") into
wave-aligned Langmuir structures? Treated as a hypothesis; the stress ratio u*/u′_rms,0 is the only varied parameter.

## Setup (what was actually run)
* WC25 configuration: unit cube, periodic x–y, impermeable free-slip top (z = 1) and bottom, WENO(9) implicit LES, RK3,
  CFL 0.5, Float64, Oceananigans 0.107 on GH200 GPUs. Deep-wave Stokes drift from the printed eq. (2.7),
  ∂z uˢ = (1/4)e^{8(z−1)} (primitive uˢ = (1/32)e^{8(z−1)}, uˢ(1) = 1/32); companion "medium" family ∂z uˢ = z/2.
  The repository driver's `DeepStokesShear(2, 8)` (2 sinh(8(z−1))/sinh(8)) is **not** eq. (2.7) (wrong amplitude, sign and
  vertical structure); it was not used.
* Initial condition: seeded, divergence-free, reflection-symmetrized random **Lagrangian** field with zero horizontal-mean
  profiles (uˢ not added, so the Eulerian mean is −uˢ); spectrum ∝ (k/k_e)⁴e^{−2(k/k_e)²}, k_e = 2π·10 (horizontal peak at
  mode 9, L₁₁ = 0.038), total rms u′_rms,0 = 0.03, K₀ = 4.5e−4, ω_rms = 2.1, Ps_deep(z) = 10 (surface) … 70 (z = 0.75). WC25's
  1000→10 vorticity spin-up was skipped by design; u′_rms,0 and k_e are pilot choices, not a WC25 match (WC25 saved data were
  not available locally).
* Stress τ = u*² as a flux boundary condition on u at z = 1 (+x, along the waves), u*/u′_rms,0 ∈ {0, 0.01, 0.03, 0.1, 0.3, 1},
  i.e. u* ∈ {0, 3e−4, 9e−4, 3e−3, 9e−3, 3e−2}, La = √(u*/uˢ(1)) ∈ {∞, 0.098, 0.17, 0.31, 0.54, 0.98}. The column's uniform
  acceleration τt is retained; the bulk velocity is removed only in diagnostics. Controls: wave-free runs from the
  **same Lagrangian field** at u*/u′_rms,0 = 0, 0.1, 1 (note: not Eulerian-matched).
* Runs: 128³ to t = 400 for all six stresses and three controls (seed 1); extensions by checkpoint pickup to t = 1000 for
  deep r ≤ 0.3 and the r = 0 control (r = 1 to t ≈ 1000, finishing); second seed at r = 0, 0.03, 1 and control 0;
  256³ at r = 0, 0.03 and 0.1; medium family r = 0, 0.1. Total cost ≈ 11 GPU-h (surface-stress campaign). Outputs: scalar statistics every 0.5, horizontal-mean
  profiles every 1, slices (x–z, y–z, x–y at three depths) at 67 times, 3-D fields at 10 times, checkpoints every 50.
* Validation (all passed): divergence-free IC with w = 0 at the walls and zero mean profiles; volume-mean momentum
  conserved to 1e−19 without stress; d⟨U⟩/dt = τ/H exactly with the +x sign; wave-only quiescent state exactly at rest;
  kinetic energy monotone under steady Stokes drift (no spurious injection); K − K₀ − ∫τ⟨u⟩_top dt ≤ 0 (implicit
  dissipation only removes energy); bitwise restart equivalence with fixed Δt.

## What the stress sequence shows
1. **Zero stress reproduces WC25 phenomenology qualitatively.** A counter-wave surface jet (U_L − ⟨U_L⟩ ≈ −3.2e−3 at
   z = 1, forming by t ≈ 20–50) and depth-alternating interior bands (amplitude 7e−4 at t = 400 → 9e−4 at t = 1000,
   wavelength ≈ 0.75 at t = 1000, 3 interior sign reversals) develop; their share of the relative energy grows,
   K_jet/K_rel = 0.19 (t = 400) → 0.40 (t = 1000), while the same ratio stays ≤ 0.02 in the wave-free control whose
   bands (2e−4) sit below its fluctuation rms. The Lagrangian mean stays nearly shear-free (U_E ≈ −uˢ persists, the
   anti-Stokes profile), w at z = 0.9 is organized into cross-wave (y-elongated) streaks, the Stokes-layer vertical
   anisotropy ⟨w²⟩/⟨u′²⟩ falls to 0.45 (control 0.7), and the along/cross fluctuation-enstrophy ratio stays ≈ 0.95.
   Energy decays more slowly with waves (K/K₀ = 1.3e−3 vs 0.9e−3 at t = 400). Seed 2 and 256³ reproduce the surface
   jet (−3.5e−3, −3.2e−3) and interior amplitude (8.8e−4, 8.4e−4) to 10–20 %; the mean-flow energy share at 256³ is
   ≈ 30 % lower. The local Rhines estimate 2π√(U/|∂zz uˢ|) with U = 7e−4 gives 0.18 at z = 0.9 and 0.32 at z = 0.75,
   the same order as the observed band spacing but not a quantitative match (the bands sit mostly where |∂zz uˢ| is small).
2. **u*/u′_rms,0 = 0.01 (La 0.10) is indistinguishable from zero stress to t = 1000**: same jets (K_jet/K_rel 0.30, 3 reversals,
   wavelength 0.77). The instantaneous ratio u*/u′_rms(t) reaches only 0.3 by t = 1000.
3. **u*/u′_rms,0 = 0.03 (La 0.17) is the transition case, and the transition is time-dependent.** It evolves like the
   zero-stress case until t ≈ 150; the along/cross enstrophy ratio exceeds 1.3 at t = 277, when u*/u′_rms(t) = 0.74, and
   reaches 1.75 by t = 400 and stays there; w skewness turns negative; by t = 1000 the Lagrangian mean is a monotone
   wind profile (no interior reversals) with U_E homogenized through the Stokes layer — the Langmuir state.
4. **u*/u′_rms,0 = 0.1 (La 0.31): Langmuir rolls from t ≈ 80** (onset when u*/u′_rms = 1.0): surface-attached y–z cells with
   narrow downwelling plumes reaching z ≈ 0.6, x-elongated w streaks, roll ratio 1.6–1.8, ⟨w²⟩/⟨u′²⟩ 1.3, w skewness −1,
   U_E uniform in the Stokes layer while U_L carries the Stokes shear, fluctuation enstrophy sustained by the wind work.
5. **u*/u′_rms,0 = 0.3 and 1 (La 0.54, 0.98): shear-dominated.** The wind layer fills the column by t ≈ 300 / 100; roll
   anisotropy never exceeds 1.2 — the same as the wave-free controls at the same stress (1.17, 1.24) — and
   ⟨w²⟩/⟨u′²⟩ ≈ 0.5 for r = 1; column-scale alternating bands appear in U_L after the layer fills but they are
   wind-layer turbulence structures, not WC25 jets. Surface velocities reach O(1) (33 u′_rms,0), where implicit WENO
   dissipation becomes velocity-dependent (Galilean-invariance caveat for these two cases).
6. **Regime rule.** Wave-aligned rolls require u*/u′_rms(t) ≳ 0.7–1 **and** La ≲ 0.3; WC25 cross-wave jets persist (and
   strengthen) while u*/u′_rms ≲ 0.3; for La ≳ 0.5 the stress produces shear turbulence with little wave organization
   whatever u*/u′_rms (r = 1 to t = 1000: roll ratio ≤ 1.18, same as its wave-free control). Because u′_rms decays while τ is
   constant, every finite stress eventually crosses the first threshold; the time of crossing is the time at which the
   decaying turbulence weakens to u′_rms ≈ u*. **Resolution test of the rule:** at 256³ the onset of rolls at r = 0.1 occurs at
   t = 156 instead of 82 (the finer run keeps more fluctuation energy early), yet the instantaneous ratio at onset is the
   same, u*/u′_rms = 1.06 vs 1.02; at r = 0.03 both resolutions give t ≈ 270–277 and u*/u′_rms = 0.66–0.74. The onset time is
   resolution-dependent, the onset ratio is not.
7. **Medium-wave companion (∂z uˢ = z/2, La defined with uˢ(1) = 1/4):** at zero stress the jets are stronger
   (surface jet 6.5e−3, interior amplitude 2.4e−3 > u′_rms 1.9e−3, K_jet/K_rel 0.34, wavelength 0.84 at t = 400; K/K₀ 5.8e−3
   vs 1.3e−3 deep), confirming that the geometry rather than the implementation controls the jet strength; at u*/u′_rms,0 = 0.1
   (La 0.11) rolls set in at t = 80 (u*/u′_rms = 0.83), the same time as for deep waves (t = 82), and the Lagrangian mean is a
   wind profile by t = 400 (surface 0.061). (Figures `medium_*.png`.)

## What remains inconclusive / caveats
* Resolution: 128³ with an energy-containing mode of 10 is strongly dissipative (K/K₀ = 2.7e−3 by t = 190). 256³ confirms
  the zero-stress mean flow (surface jet −3.21e−3 vs −3.24e−3) though K_jet/K_rel is 30 % lower; the transition case r = 0.03
  at 256³ has a 20 % stronger wind layer (surface 1.33e−2 vs 1.11e−2) and a slightly earlier roll onset (⟨ω′ₓ²⟩/⟨ω′ᵧ²⟩ ≈ 1.2 at
  t = 400 in the raw measure); the Langmuir case r = 0.1 at 256³ matches 128³ to within 10 % in u′_rms, u′, surface mean and K.
  [Correction 2026-10-03, classifier audit: the 1.2 quoted here is the raw volume ratio ⟨ω_x²⟩_V/⟨ω_y²⟩_V of the total
  vorticity, which includes the mean shear; the primes are wrong. The fluctuation ratio ⟨ω′ₓ²⟩/⟨ω′ᵧ²⟩ (mean shear removed,
  depth mean over all faces) is 1.658 at 256³ and 1.757 at 128³ at t = 400, with onset t = 270 and 277; the onset is slightly
  earlier at 256³ (270 versus 277). The 128³ and 256³ ICs are different realizations. See REPORT_resolution.md.]
* The jets here are 3–10× weaker than WC25's (which start from a spun-up, larger-scale IC) and the band structure is
  dominated by 2–3 bands in a unit box; a domain-sensitivity run (wider/deeper box) was not done.
* Only one seed at r = 0.01, 0.1, 0.3; the onset time at r = 0.03 (277) is from one seed.
* La ≥ 0.5 cases exceed CFL-safe velocities for the implicit LES to be velocity-independent; a Galilean-shifted
  check was not run.
* The energy budget uses the discrete kinetic energy with wind work τ⟨u⟩_top; the budget residual is the implicit
  dissipation, consistent in sign at all times, but no spectral transfer was computed, so no claim about inverse-cascade
  direction is made.

## Next most informative runs
1. 512³ (or 256³ to t = 1000) at u*/u′_rms,0 = 0.03 to pin the onset time; the 256³ runs done here support the 128³ picture.
2. A second seed at r = 0.01 and 0.1, and r = 0.02/0.05 to tighten the threshold u*/u′_rms ≈ 0.7–1 and La ≈ 0.3.
3. A larger IC scale (mode 6) or a WC25-style spun-up IC at 256³ to test whether the jet strength and Rhines scaling
   approach WC25's, and a 2×2×1 domain to test box-scale confinement of the bands.
4. A Galilean-frame check for r ≥ 0.3.

Figures: `figures/wc25_surface_stress/{pilot1,sweep1,sweep2,robust,medium}_*.png`; data root
`/work/hdd/bhcr/glwagner/wc25_surface_stress_2026-10-02`; job/cost table `wc25_surface_stress_coordination/ledger/case_table.md`.
