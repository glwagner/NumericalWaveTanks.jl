# WC25 main-figure resolution audit (2026-10-04)

Requirement (Greg, clarification 2, 2026-10-04): every main gallery/results plot uses the **highest completed suitable
resolution**, 1024² where it exists; lower resolutions appear only in explicit comparisons against it; matched
case, physics and times; per-panel provenance; the 512³ regime target stays **incomplete** until coverage exists.
Publication base: glw/anti-stokes c3a9317. Status: **NOT COMPLETE** — 2-D main figures replaced at 1024² (jobs 3307685, 3307758; provenance PASS); 3-D main figures at 256³
(jobs 3307725, 3307758); 512³ r = 0.1 and 0.03 continuation jobs queued, r = 0 queued; 512³ regime coverage partial by budget.

## A. Vallis 2-D (section B of the gallery and results document)

Highest completed grid for every main 2-D case is **1024²**: ring-9 (k₀ = 9, σ = 1, rb = 1, seed 1) shallow / deep / none
to 200 T_e, and the k₀ = 12, rb = 2 ensemble (shallow seeds 1–6, none seeds 1–3) to 60 T_e. All COMPLETED.

| published plot (9c4b59d) | published grid | replacement | grid | status |
|---|---|---|---|---|
| final_ic_heatmap | 256² | main1024_ic_heatmap | 1024² | done (3307685) |
| final_panels_shallow | 256² | main1024_panels_shallow | 1024² | done |
| final_spectra_shallow_heatmap (Rhines dashed) | 256² | main1024_spectra_shallow_heatmap, **no overlay** | 1024² | done |
| final_spectra_deep_heatmap | 256² | main1024_spectra_deep_heatmap | 1024² | done |
| (final_spectra_none_heatmap, linked) | 256² | main1024_spectra_none_heatmap | 1024² | done |
| final_comparison_t200 | 256² | main1024_comparison_t200 | 1024² | done |
| final_timeseries | 256² + β-plane 256² | main1024_timeseries; β-plane reference curve labelled "256²" | 1024² (+ ref. 256²) | done |
| final_transfer | 256² | main1024_transfer | 1024² | done |
| final_jets | 256² | main1024_jets | 1024² | done |
| final_ensemble_spectra_shallow_heatmap (Rhines dashed) | 256² | main1024_ensemble_spectra_shallow_heatmap, no overlay | 1024² | done |
| final_ensemble_spectra_none_heatmap | 256² | main1024_ensemble_spectra_none_heatmap | 1024² | done |
| final_betaplane_heatmap | 256² both | keep as an explicit **matched 256² comparison** (the doubly periodic reference exists only at 256²) | 256² | flag: 512² reference ≈ 0.7 GPU-h (2-D), 1024² ≈ 5 GPU-h (exceeds 2-D headroom) |
| final_variants (rb/σ/λ/zonal/seed bracket) | 256² | keep as an explicit **256² sensitivity comparison** (variants exist only at 256²) | 256² | flag: 1024² variants ≈ 10 runs × ≈ 0.1 = ≈ 1 GPU-h (2-D) — not run, needs approval |
| res3v2_* (ring-9, no overlays) | 256²/512²/1024² | explicit three-grid comparison, replaces res3 (which drew Rhines curves) | — | done (3307758) |
| ens12_spectra_{shallow,none}_heatmap | 256²/512²/1024² | unchanged: explicit three-grid comparison | — | keep |
| res3_* | 256²/512²/1024² | unchanged: explicit three-grid comparison | — | keep |

Provenance gate (analysis/wc25_vallis2d/main_provenance.jl, runs first in 3307685; figures refuse to draw without PASS):
identical physical parameters to the published 256² case, same initial condition (t = 0 spectral density on the low-mode
window, max|ΔD|/max D < 5e−3; continuous coefficients shared by construction), COMPLETED marker, and every panel time
(0/20/40/200 T_e; ensembles 0/20/60 T_e) present within 0.5 T_e. Output: figures/wc25_vallis2d/main1024_provenance.txt.

## B. Surface stress 3-D (section A)

**Update 2026-10-06 (later).** 512³ r = 0.03 COMPLETED (t = 400.02, verified). Main figures are now `mainmix2_*`: r = 0 and 0.03 at 512³, r = 0.1
and 0.3 at 256³ (mixed, each labelled). New matched comparison `res3_r0p03_512_*` (128³ / 256³ / 512³): the r = 0.03 onset time moves from 270–273
to 343 at 512³ at a similar onset ratio. Still running: 512³ r = 0.1 (t ≈ 150 → 400) and 256³ wave-free r = 0.3.

**Update 2026-10-06.** Completed since: 512³ r = 0 (t = 400, outputs verified) and 256³ deep r = 0.3 (t = 400). Main figures are now the
per-case highest completed grid (`mainmix_*`: r = 0 at 512³; r = 0.03, 0.1, 0.3 at 256³), each case labelled with its grid; the all-256³ set
(`main256v2_*`) and the matched 128³/256³/512³ r = 0 comparison (`res3_r0_512_*`) are labelled comparisons. Still running: 512³ r = 0.03 (t ≈ 376 → 400)
and r = 0.1 (t ≈ 129 → 400); 256³ wave-free r = 0.3 pending. Optional extra 512³ cases deferred by review (storage).


Completed grids: 128³ legacy draw (all r, wave-free, medium; t ≤ 1000 for several), 128³ d256 (r = 0, 0.03, 0.1, t = 400;
matched to the 256³ draw), **256³: deep r = 0, 0.1 (t = 400) and r = 0.03 (t = 1000)**, 256³-draw seed-1 IC.
No completed 512³ run. No wave-free, r = 0.01, r = 0.3 or r = 1 case above 128³ (256³ r = 0.3 deep + wave-free queued).

| published plot | published grid/cases | highest completed | action |
|---|---|---|---|
| final_regime | 128³ legacy, deep r = 0…1 + wave-free, t ≤ 1000 | 256³ for r = 0, 0.03, 0.1 only | main256_regime (done) for the three r; 128³ sweep kept only as a labelled coverage comparison; 512³ replaces 256³ when complete |
| final_energy, final_anisotropy, final_jets | 128³ legacy | 256³ (three r) | main256_* (done), same treatment |
| final_sections_t400 | 128³ legacy, 9 cases | 256³ (three r) at t = 100, 400 | main256_sections_t{100,400} (done) |
| final_xz_spectra_heatmap | 128³ legacy, t = 0/400/1000 | 256³ (three r), t = 0/100/400 | main256_xz_spectra_heatmap (done); t = 1000 only exists at 256³ for r = 0.03 so it is not shown (no substituted time) |
| jets_vs_rolls_sections.mp4 (listed in results file table) | 128³ | 256³ (three r) | flag: re-render with animate_sections.jl after 512³ completes |
| frame128_frame_check | 128³ comparison | — | explicit comparison; 256³ frame check optional if budget remains |

Caveat that must appear in captions: the 256³ (and 512³) runs use the **256³-draw** seed-1 realization; the published
128³ legacy seed-1 runs are a *different realization of the same statistics*. The matched lower-resolution baseline for
any 256³/512³ panel is the 128³ d256 run (same coefficients), not the legacy 128³ run.

## C. 512³ regime coverage and billed-cost estimate

Cost model: 1.20 s/iteration at 512³ (ic_bench512 benchmark: 745 it in 14.85 min), iterations ≈ 4–5.5 × the 128³ count
(128³→256³ ratios measured 1.15 at r = 0 and 2.35 at r = 0.1), + ≈ 6 min per 2 h segment.

| 512³ case (deep unless noted), t = 400 | status | est. billed GPU-h |
|---|---|---|
| r = 0 | queued 3304509 | 2.5–4 |
| r = 0.03 | continuation queued 3308264 (first segment 3301574 completed) | 3–5 |
| r = 0.1 | continuation queued 3308232 (first segment 3301573 completed) | 11–15 |
| r = 0.01 | missing | 2.5–4 |
| wave-free r = 0 | missing | 2.5–4 |
| medium r = 0 (separate family figure) | missing | 2.5–4 |
| wave-free r = 0.1 | missing | 11–15 |
| medium r = 0.1 | missing | 11–15 |
| r = 0.3 | missing | ≈ 40–50 |
| r = 1 | missing | ≈ 150–200 |
| wave-free r = 1 | missing | ≈ 170–220 |
| extension of any case to t = 1000 | missing | ≈ 1.5 × its t = 400 cost again |

Budget: 60 GPU-h total; billed 18.089 at 15:25 UTC (incl. 12.9 prior); projected after all queued work 38–49 → headroom **11–22 GPU-h**.
**Full 512³ coverage of the published regime diagram cannot fit the 60 GPU-h ceiling (rough estimate ≈ 420 GPU-h to t = 400; budget coverage of the sweep is incomplete).**
Feasible additions, in priority order, each submitted only when a slot is free and the worst-case projection stays ≤ 60:
1. deep r = 0.01, 512³, t = 400 (≈ 2.5–4) — extends the deep sweep toward the jet regime;
2. wave-free r = 0, 512³, t = 400 (≈ 2.5–4) — control;
3. medium r = 0, 512³ (≈ 2.5–4) — only if the measured 512³ r = 0.1 cost leaves room.
Not feasible under the ceiling: r = 0.3, r = 1, wave-free r = 0.1/1, medium r = 0.1, any t = 1000 extension.
A 512³ regime diagram will therefore contain r = 0, 0.01, 0.03, 0.1 (+ wave-free r = 0) only, labelled as partial coverage,
with no 128³/256³ points mixed in.

## D. Outside this assignment
The anti-Stokes sections of gallery.html (cases 1.A–1.D, M0/M2 videos) belong to another campaign; not audited here.
Case 1.D's first video is M0 (768 × 64 × 64) while M2 exists for other 1.D views — flagged for that campaign's owner.
