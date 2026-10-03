| Job | Label | Submitted (UTC) | State | Elapsed | GPU-h | Members / notes |
|---|---|---|---|---|---|---|
| 3296238 | validate | 2026-10-02 21:52:19Z | FAILED | 00:02:18 | 0.04 | --time=01:30:00 batch/wc25_validate.batch N=64 N_pilot=128 T=10 |
| 3296263 | validate | 2026-10-02 21:56:13Z | FAILED | 00:26:10 | 0.44 | --time=01:30:00 batch/wc25_validate.batch N=64 N_pilot=128 T=10 |
| 3296924 | validate2 | 2026-10-02 22:25:19Z | FAILED | 00:15:02 | 0.25 | --time=01:00:00 batch/wc25_validate.batch N=64 N_pilot=128 T=10 steps=restart,timing |
| 3296925 | ic_N128_seed1 | 2026-10-02 22:25:19Z | COMPLETED | 00:01:25 | 0.02 | --time=00:30:00 batch/wc25_generate_ic.batch N=128 seed=1 q0=0.03 mode=10 |
| 3296928 | pilot_A_deep0_none0 | 2026-10-02 22:27:06Z | COMPLETED | 00:22:25 | 0.37 | --time=04:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3296929 | pilot_B_deep0p1 | 2026-10-02 22:27:07Z | COMPLETED | 00:11:52 | 0.20 | --time=03:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3296930 | pilot_C_deep1_none1 | 2026-10-02 22:27:07Z | COMPLETED | 01:00:31 | 1.01 | --time=04:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3297002 | validate3_restart | 2026-10-02 22:40:07Z | COMPLETED | 00:07:59 | 0.13 | --time=00:40:00 batch/wc25_validate.batch N=64 T=10 steps=restart |
| 3297005 | analysis_test_N64 | 2026-10-02 22:40:50Z | FAILED | 00:00:36 | 0.01 | --time=00:45:00 batch/wc25_analysis.batch script=compare_stress.jl root=/work/hdd/bhcr/glw |
| 3297015 | analysis_test_N64 | 2026-10-02 22:42:16Z | FAILED | 00:00:27 | 0.01 | --time=00:45:00 batch/wc25_analysis.batch script=compare_stress.jl root=/work/hdd/bhcr/glw |
| 3297025 | analysis_test_N64 | 2026-10-02 22:43:21Z | FAILED | 00:00:52 | 0.01 | --time=00:45:00 batch/wc25_analysis.batch script=compare_stress.jl root=/work/hdd/bhcr/glw |
| 3297029 | analysis_test_N64 | 2026-10-02 22:44:49Z | COMPLETED | 00:01:02 | 0.02 | --time=00:45:00 batch/wc25_analysis.batch script=compare_stress.jl root=/work/hdd/bhcr/glw |
| 3297046 | sweep_D_deep0p01_0p03 | 2026-10-02 22:47:33Z | COMPLETED | 00:20:47 | 0.35 | --time=02:30:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3297056 | sweep_E_deep0p3_none0p1 | 2026-10-02 22:49:11Z | COMPLETED | 00:30:09 | 0.50 | --time=02:30:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3297064 | analysis_pilot1 | 2026-10-02 22:50:07Z | COMPLETED | 00:01:19 | 0.02 | --time=01:00:00 batch/wc25_analysis.batch script=compare_stress.jl cases=deep_r0_seed1_N12 |
| 3297086 | ext1000_deep0_none0_deep0p1 | 2026-10-02 22:53:03Z | COMPLETED | 01:25:39 | 1.43 | --time=02:30:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3297193 | analysis_sweep1 | 2026-10-02 23:09:13Z | COMPLETED | 00:02:30 | 0.04 | --time=01:00:00 batch/wc25_analysis.batch script=compare_stress.jl cases=deep_r0_seed1_N12 |
| 3297205 | ext1000_deep0p01_0p03_0p3_1 | 2026-10-02 23:12:22Z | COMPLETED | 02:42:32 | 2.71 | --time=03:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3297246 | ic_N128_seed2 | 2026-10-02 23:20:30Z | COMPLETED | 00:01:26 | 0.02 | --time=00:30:00 batch/wc25_generate_ic.batch N=128 seed=2 q0=0.03 mode=10 |
| 3297258 | seed2_deep0_0p03_none0_deep1 | 2026-10-02 23:22:40Z | COMPLETED | 00:59:33 | 0.99 | --time=03:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=2 N=128 st |
| 3297296 | ic_N256_seed1 | 2026-10-02 23:27:51Z | COMPLETED | 00:01:23 | 0.02 | --time=00:45:00 batch/wc25_generate_ic.batch N=256 seed=1 q0=0.03 mode=10 |
| 3297338 | N256_deep0 | 2026-10-02 23:31:07Z | COMPLETED | 00:20:30 | 0.34 | --time=06:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=256 st |
| 3297449 | N256_deep0p03 | 2026-10-03 00:00:28Z | COMPLETED | 00:22:55 | 0.38 | --time=06:00:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=256 st |
| 3297519 | vallis_v01_validate | 2026-10-03 00:19:40Z | FAILED | 00:01:41 | 0.03 | --time=00:50:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3297553 | medium_r0_r0p1 | 2026-10-03 00:23:04Z | COMPLETED | 00:23:21 | 0.39 | --time=02:30:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=128 st |
| 3297593 | vallis_v01b_validate | 2026-10-03 00:33:43Z | FAILED | 00:03:59 | 0.07 | --time=00:50:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3297610 | vallis_v02_pilots | 2026-10-03 00:39:43Z | FAILED | 00:50:10 | 0.84 | --time=01:00:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3297735 | analysis_all_1 | 2026-10-03 01:16:58Z | COMPLETED | 00:05:59 | 0.10 | --time=01:00:00 batch/wc25_analysis_all.batch |
| 3297789 | vallis_v02b_leftovers | 2026-10-03 01:30:24Z | COMPLETED | 00:08:51 | 0.15 | --time=01:00:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3297811 | animations_1 | 2026-10-03 01:33:43Z | FAILED | 00:01:09 | 0.02 | --time=00:40:00 batch/wc25_analysis.batch script=animate_sections.jl cases=deep_r0_seed1_N |
| 3297931 | vallis_v03_512 | 2026-10-03 01:56:17Z | COMPLETED | 00:11:38 | 0.19 | --time=00:50:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3297961 | animations_2 | 2026-10-03 02:01:44Z | COMPLETED | 00:00:37 | 0.01 | --time=00:40:00 batch/wc25_analysis.batch script=animate_sections.jl cases=deep_r0_seed1_N |
| 3297970 | N256_deep0p1 | 2026-10-03 02:03:06Z | COMPLETED | 00:49:25 | 0.82 | --time=01:30:00 /u/glwagner/nwt_wc25_surface_stress/batch/wc25_multi.batch seed=1 N=256 st |
| 3297996 | vallis_v02c_none_figures | 2026-10-03 02:12:17Z | COMPLETED | 00:08:36 | 0.14 | --time=00:40:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3298003 | vallis_v03b_betaplane | 2026-10-03 02:14:18Z | COMPLETED | 00:04:21 | 0.07 | --time=00:40:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3298023 | vallis_v04_sensitivity | 2026-10-03 02:20:23Z | COMPLETED | 00:14:41 | 0.24 | --time=01:15:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3298024 | vallis_v05_ensemble | 2026-10-03 02:21:23Z | COMPLETED | 00:13:38 | 0.23 | --time=01:15:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3298222 | analysis_final | 2026-10-03 03:33:22Z | COMPLETED | 00:08:01 | 0.13 | --time=01:00:00 batch/wc25_analysis_final.batch |
| 3298711 | vallis_v06_final_figures | 2026-10-03 03:49:38Z | COMPLETED | 00:02:33 | 0.04 | --time=00:30:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3298719 | analysis_final256_labels | 2026-10-03 03:50:45Z | COMPLETED | 00:01:56 | 0.03 | --time=00:30:00 batch/wc25_analysis.batch script=compare_stress.jl cases=deep_r0_seed1_N25 |
| 3298790 | vallis_v07_ensemble_figures | 2026-10-03 04:16:28Z | COMPLETED | 00:03:11 | 0.05 | --time=00:30:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |
| 3298804 | vallis_v08_figures_cosmetic | 2026-10-03 04:20:53Z | COMPLETED | 00:03:27 | 0.06 | --time=00:25:00 --chdir=/u/glwagner/nwt_wc25_vallis2d /u/glwagner/nwt_wc25_vallis2d/batch/ |

Total billed GPU-hours: 12.94 (of which Vallis-2D: 2.11; ceiling 60, Vallis sub-budget 8).
