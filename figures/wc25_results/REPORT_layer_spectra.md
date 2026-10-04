# Partial-depth spectra: is a deep-wave dumbbell hidden in the full-depth average? (2026-10-04)

**PROVISIONAL — convergence not established (1024²).** Authorized by Greg, 2026-10-04. Script `analysis/wc25_vallis2d/layer_spectra.jl`
(branch glw/wc25-resolution), Slurm jobs 3307782 and 3307795, run on the existing 1024² ring-9 runs (k₀ = 9, σ = 1, rb = 1, seed 1;
deep, shallow, wave-free; all COMPLETED to 200 T_e). No simulation was rerun. Raw numbers: `figures/wc25_vallis2d/layer1024_metrics.txt`.
We tested for a dumbbell; we did not assume one.

## Outcome in one paragraph
**Partial and ambiguous.** In the upper layer where β_eff is largest, the deep-wave eddy spectrum becomes more anisotropic than its
full-depth average and concentrates energy in lobes near the k_z axis (|k_x| ≲ 3, k_z ≈ ±4–6 cycles/box), at the same place as the
shallow-wave lobes. This signal is robust to the layer cutoff and taper and is absent from the mirrored bottom layer and from the
wave-free run. But the feature that defines a dumbbell, a depleted region around the k_x axis, is not clearly present: the fraction of
eddy energy within 30° of the k_x axis stays near 0.25, three times the shallow-wave value under the identical window (0.08) and above
the window's own baseline on the isotropic initial field (0.19). The layer is thin, so the vertical window smears k_z by about
±1 cycle/box and could partly fill a narrow hole; β_eff also varies tenfold inside the layer, so no single dumbbell scale exists there.
We therefore do not claim a hidden dumbbell, and we do not rule one out.

## Method (fixed before looking at the layer spectra)
- **Layer from the prescribed profile.** `experiments/wc25_vallis2d/common.jl` (DeepStokes) sets β_eff(z) = 2e^{8(z−1)} on z ∈ [0, 1],
  surface at z = 1 (verified). β_eff ≥ θ β_eff(1) gives 1 − z ≤ h(θ) = ln(1/θ)/8: **primary θ = 0.1, h = 0.288**; sensitivities
  θ = 0.2 (h = 0.201) and 0.05 (h = 0.374). This is an operational choice, not a sharp physical boundary.
- **Control.** A bottom layer of the same thickness and the same (mirrored) window: identical bandwidth and wall type, negligible β_eff.
  The same windows are applied to the shallow (uniform β = ½, positive control) and wave-free (negative control) runs.
- **Velocities, not ψ × mask.** u (x-faces, z-centres) and w (x-centres, z-faces) are each multiplied by a vertical taper W(z) at their
  own points. Multiplying ψ by a mask and differentiating would create a spurious edge velocity, and is not done.
- **Taper.** W = 1 from the real free-slip wall (surface, or bottom for the control) to a cosine roll-off that reaches zero at the
  artificial cut; roll-off width α h with **α = 0.5** primary and 0.25, 1.0 (half-Hann) as sensitivities. Nothing is imposed at the cut,
  in particular no sine-wall condition there. The flat part keeps the real boundary condition (∂z u = 0, w = 0 at the wall) intact.
- **Transform and units.** Fourier in x and the wall-consistent bases in z on the **full domain** (DCT-II for u at centres, DST-I for
  w at interior faces), so k_x/2π is an integer and k_z/2π = n/2 exactly as in the main spectra. With W ≡ 1 this reproduces the
  published ψ-based spectrum mode by mode (max difference 1.6e−9 of the peak). The windowed field is zero below the layer; that is the
  window, not zero padding, and adds no resolution.
- **What the window does.** The layer spectrum equals the full-domain spectrum convolved with the window's own transform. Effective
  vertical bandwidth (half-power half-width of |Ŵ(k_z)|²): **1.0 cycle/box** for the primary window, 1.0–2.0 across the sensitivity
  windows, against the native bin spacing 0.5. The window response has side lobes near 3 cycles/box at the 1e−2 to 3e−2 level; these
  produce the vertical streaks visible in the layer heatmaps. Features narrower than about 2 cycles/box in k_z cannot be resolved.
- **Normalization.** Σ E_W = ½[Σ(W u)² + Σ(W w)²]/N² exactly (Parseval; checked for every window and time to 1e−10). Dividing by
  ⟨W²⟩_z gives the mean energy density of the weighted layer; plotted is E_W/(⟨W²⟩K₀) per (cycle/box)² with K₀ the full-domain initial
  energy, on one log₁₀ colour scale for every family, window and time (eddy figures −4.70 … −0.70), grey below a four-decade floor.
- **Total and eddy.** Total layer spectra include the k_x = 0 column (the x-mean at each depth). Eddy spectra exclude it, i.e. the
  horizontal mean is removed; the exact-zonal fraction is reported separately.
- **Diagnostics.** Eddy anisotropy (⟨k_z²⟩−⟨k_x²⟩)/(⟨k_z²⟩+⟨k_x²⟩); polar and equatorial fractions = eddy energy within 30° of the
  k_z and k_x axes in the band 1 ≤ K ≤ 8 cycles/box (below the ring, where a dumbbell would form); exact-zonal fraction.
  **No Rhines or other theoretical curve is drawn.**

## Checks
| check | result |
|---|---|
| Parseval, every window and time | Σ E_W equals the weighted physical KE to < 1e−10 |
| Full-depth equivalence (W ≡ 1) | velocity spectrum = published ψ spectrum, max difference 1.6e−9 of the peak; Σ E = K(t) to 1e−10 |
| Window bias on the isotropic initial field (full depth: anisotropy +0.004, polar 0.30, equatorial 0.28) | upper primary +0.11 / 0.28 / 0.19; bottom +0.07 / 0.31 / 0.08; all windows +0.07 … +0.18 |
| Leakage out of the ring on the initial field (energy outside 6–12 cycles) | 0.002 full depth; 0.006–0.046 windowed |
| Thin-layer sampling of one realization | initial layer KE/K₀ = 1.23 (upper) vs 0.83 (bottom): thin layers of a single field are not statistically equivalent |

## Results (eddy anisotropy / polar / equatorial; exact-zonal fraction of the total)
| 200 T_e | full depth | upper layer (θ = 0.1) | bottom control |
|---|---|---|---|
| deep | +0.40 / 0.34 / 0.37; 0.45 | **+0.48 / 0.52 / 0.25; 0.48** | +0.05 / 0.11 / 0.27; 0.49 |
| shallow (positive control) | +0.66 / 0.81 / 0.04; 0.05 | +0.50 / 0.50 / 0.08; 0.15 | +0.76 / 0.88 / 0.02; 0.03 |
| wave-free (negative control) | −0.09 / 0.16 / 0.46; 0.15 | −0.07 / 0.25 / 0.29; 0.06 | −0.04 / 0.17 / 0.40; 0.09 |

At 40 T_e the deep upper layer already gives +0.40 / 0.46 / 0.20 (full depth +0.24 / 0.41 / 0.24; bottom −0.04 / 0.21 / 0.40).
Sensitivity (deep, 200 T_e, nine windows θ ∈ {0.2, 0.1, 0.05} × α ∈ {0.25, 0.5, 1}): eddy anisotropy 0.48–0.51 (window baselines
0.07–0.18), polar 0.46–0.72, equatorial 0.11–0.32. The polar fraction rises and the equatorial fraction falls for thinner layers and
for the half-Hann taper, which largely coincide with the widest window response; so the most dumbbell-like numbers come from the windows
that smear k_z most, and should not be read as the cleanest estimate.

## Interpretation and limitations
- The deep-wave upper layer carries 2.7 times the initial domain-mean energy density at 200 T_e (layer KE/K₀ = 2.67), almost half of it in the
  exact-zonal surface jet (0.48). The eddy part of that layer is anisotropic with shallow-like polar lobes. Against the same window,
  the shallow upper layer reaches about the same anisotropy (+0.50) and polar fraction (0.50), but a much smaller equatorial fraction (0.08).
- Even the uniform-β shallow case loses much of its dumbbell signature in a thin window (full depth +0.66 / 0.81 → upper +0.50 / 0.50),
  and its upper and bottom layers differ by 0.26 in anisotropy. Differences of that size between layers therefore come from window
  smearing and single-realization sampling alone. The deep upper-minus-bottom difference (0.43) exceeds it; the deep-minus-shallow
  difference in the equatorial fraction (0.25 vs 0.08) is the main evidence against a clear hidden dumbbell.
- β_eff falls from 2 at the surface to 0.2 at the primary cut, so even within the layer a constant-β dumbbell scale does not exist.
- Single realization, one grid (1024², convergence not established), three saved times for the heatmaps and every saved field
  (2.5 T_e spacing) for the time series.

## Figures (figures/wc25_vallis2d/)
- `layer1024_deep_eddy.png` (main): deep, rows full depth / upper layer / bottom control, columns 0 / 40 / 200 T_e, eddy spectra.
- `layer1024_compare_t200.png`: 200 T_e, rows deep / shallow / wave-free, columns full / upper / bottom; identical windows and colours.
- `layer1024_diagnostics.png`: windows and β_eff profile, window response (bandwidth), exact-zonal fraction, eddy anisotropy and
  polar − equatorial against time for every family and window, with the initial-field window baseline.
- `layer1024_sensitivity_deep_t200.png`: deep, 200 T_e, nine windows (θ × α).
- `layer1024_{deep,shallow,none}_{eddy,total}.png`: all families, eddy and total.
