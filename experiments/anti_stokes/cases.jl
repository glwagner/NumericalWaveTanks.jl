#####
##### Experimental cases from Ellingsen et al. (2026), "Turbulence-induced
##### anti-Stokes flow: experiments and theory", JFM 1029, A6, Tables 2 and 3.
#####
##### Only directly reported quantities are stored; everything else is derived
##### in one place so that no rounded intermediate values are scattered around.
##### Water depth h = 0.40 m for all of Experiment 1. The intrinsic group duration
##### τ₀ is the paper's tabulated value (Table 3).
#####

"""
    wave_group_case(FT; name, U₀, k, τ_lab, τ₀, u_rms, v_rms, w_rms, e, L, steepness, h=0.40)

Build the NamedTuple describing one wave-group case of Experiment 1: the measured
quantities plus the derived phase speed `c`, group speed `cᵍ`, surface Stokes drift
`Uˢ₀ = ϵ² c`, intrinsic spatial half-width `σ₀ = cᵍ τ₀`, Stokes e-folding depth
`δˢ = 1 / 2k`, and effective interaction time `Tᵢ = √π τ₀`.
"""
function wave_group_case(FT=Float32; name, U₀, k, τ_lab, τ₀, u_rms, v_rms, w_rms, e, L, steepness, h=0.40)
    g = FT(9.81)

    measured = (; name,
                  h         = FT(h),          # water depth [m]
                  U₀        = FT(U₀),         # mean current magnitude [m s⁻¹]
                  k         = FT(k),          # carrier wavenumber [m⁻¹]
                  τ_lab     = FT(τ_lab),      # laboratory group duration [s]
                  τ₀        = FT(τ₀),         # intrinsic group duration [s]
                  u_rms     = FT(u_rms),      # streamwise rms velocity [m s⁻¹]
                  v_rms     = FT(v_rms),      # spanwise rms velocity [m s⁻¹]
                  w_rms     = FT(w_rms),      # vertical rms velocity [m s⁻¹]
                  e         = FT(e),          # turbulent kinetic energy [m² s⁻²]
                  L         = FT(L),          # streamwise integral scale Lₓˣ [m]
                  steepness = FT(steepness))  # peak steepness ϵ = k₀ aₚ

    c   = sqrt(g / measured.k)            # deep-water phase speed
    cᵍ  = c / 2                           # deep-water group speed
    Uˢ₀ = measured.steepness^2 * c        # surface Stokes drift at packet peak
    σ₀  = cᵍ * measured.τ₀                # intrinsic spatial amplitude width
    δˢ  = 1 / (2 * measured.k)            # Stokes e-folding depth
    Tᵢ  = sqrt(FT(π)) * measured.τ₀       # effective interaction time

    return merge(measured, (; g, c, cᵍ, Uˢ₀, σ₀, δˢ, Tᵢ))
end

"""
    case_1A(FT=Float32)

Case 1.A: stationary grid (weakest, smallest-scale turbulence).
"""
case_1A(FT=Float32) = wave_group_case(FT; name="1.A", U₀=0.34, k=9.5, τ_lab=7.3, τ₀=2.4,
                                          u_rms=0.0071, v_rms=0.0068, w_rms=0.0058, e=0.65e-4,
                                          L=0.051, steepness=0.20)

"""
    case_1B(FT=Float32)

Case 1.B: only the vertical bars of the active grid actuated.
"""
case_1B(FT=Float32) = wave_group_case(FT; name="1.B", U₀=0.33, k=9.2, τ_lab=7.7, τ₀=2.6,
                                          u_rms=0.011, v_rms=0.010, w_rms=0.0073, e=1.3e-4,
                                          L=0.26, steepness=0.20)

"""
    case_1C1(FT=Float32)

Case 1.C.1: active grid, the weaker packet of the steepness pair (ϵ = 0.15, k₀ = 8.9 m⁻¹).
"""
case_1C1(FT=Float32) = wave_group_case(FT; name="1.C.1", U₀=0.33, k=8.9, τ_lab=7.9, τ₀=2.9,
                                           u_rms=0.016, v_rms=0.012, w_rms=0.0092, e=2.4e-4,
                                           L=0.32, steepness=0.15)

"""
    case_1C2(FT=Float32)

Case 1.C.2: same turbulence as 1.C.1, the steeper packet (ϵ = 0.22, k₀ = 9.0 m⁻¹).
"""
case_1C2(FT=Float32) = wave_group_case(FT; name="1.C.2", U₀=0.33, k=9.0, τ_lab=7.9, τ₀=2.9,
                                           u_rms=0.016, v_rms=0.012, w_rms=0.0092, e=2.4e-4,
                                           L=0.32, steepness=0.22)

"""
    case_1D(FT=Float32)

Case 1.D: the wave-group case with the strongest turbulence-induced anti-Stokes signal.
"""
case_1D(FT=Float32) = wave_group_case(FT; name="1.D", U₀=0.34, k=9.3, τ_lab=7.8, τ₀=2.8,
                                          u_rms=0.017, v_rms=0.017, w_rms=0.013, e=3.7e-4,
                                          L=0.20, steepness=0.22)

#####
##### Regular-wave cases (Experiments 2 and 3, Tables 1–3). Planar PIV measured u and w only,
##### so v_rms is taken equal to u_rms; the integral scale was not reported for Experiment 2
##### and is assumed to be 0.15 m there (flagged by `L_assumed`). The current-following frame
##### sees a steady, horizontally uniform Stokes drift acting on turbulence that has decayed
##### since the grid; the field of view corresponds to t_FOV = L_FOV / U₀ after the grid.
#####

"""
    regular_wave_case(FT; name, family, h, U₀, k, u_rms, w_rms, e, L, steepness, L_FOV=8.5, v_rms=u_rms, L_assumed=false)

One steepness of a regular-wave case. `family` names the turbulence state (e.g. `"2.A"`) whose
checkpoints are shared by all wavenumbers and steepnesses of that state. Derived: `c`, `Uˢ₀`,
`δˢ`, `t_FOV = L_FOV / U₀`, and the closed-channel Eulerian return flow
`u_rf = −Uˢ₀ (1 − e^{−2kh}) / (2kh)` that Table 3 tabulates.
"""
function regular_wave_case(FT=Float32; name, family, h, U₀, k, u_rms, w_rms, e, L, steepness,
                           L_FOV=8.5, v_rms=u_rms, L_assumed=false)
    g = FT(9.81)
    measured = (; name, family, h = FT(h), U₀ = FT(U₀), k = FT(k), u_rms = FT(u_rms), v_rms = FT(v_rms),
                  w_rms = FT(w_rms), e = FT(e), L = FT(L), steepness = FT(steepness), L_FOV = FT(L_FOV))
    c    = sqrt(g / measured.k)
    Uˢ₀  = measured.steepness^2 * c
    δˢ   = 1 / (2 * measured.k)
    t_FOV = measured.L_FOV / measured.U₀
    u_rf = -Uˢ₀ * (1 - exp(-2 * measured.k * measured.h)) / (2 * measured.k * measured.h)
    return merge(measured, (; g, c, Uˢ₀, δˢ, t_FOV, u_rf, regular = true, L_assumed))
end

const REGULAR_WAVE_FAMILIES = Dict(
    # family => (h, U₀, u_rms, w_rms, e, L, L_assumed)
    "2.A" => (h = 0.80, U₀ = 0.30, u_rms = 0.025,  w_rms = 0.018,  e = 6.4e-4,  L = 0.15,  L_assumed = true),
    "2.B" => (h = 0.80, U₀ = 0.30, u_rms = 0.016,  w_rms = 0.012,  e = 2.7e-4,  L = 0.15,  L_assumed = true),
    "3.A" => (h = 0.50, U₀ = 0.19, u_rms = 0.012,  w_rms = 0.0084, e = 1.4e-4,  L = 0.054, L_assumed = false),
    "3.B" => (h = 0.50, U₀ = 0.19, u_rms = 0.0087, w_rms = 0.0070, e = 0.87e-4, L = 0.051, L_assumed = false))

# name => (family, k, steepness)
const REGULAR_WAVE_TABLE = (
    ("2.A.1.1", "2.A", 6.1,  0.09), ("2.A.1.2", "2.A", 6.1,  0.14), ("2.A.1.3", "2.A", 6.1,  0.18),
    ("2.A.2.1", "2.A", 12.1, 0.15), ("2.A.2.2", "2.A", 12.1, 0.19), ("2.A.2.3", "2.A", 12.1, 0.21),
    ("2.B.1.1", "2.B", 6.1,  0.07), ("2.B.1.2", "2.B", 6.1,  0.12), ("2.B.1.3", "2.B", 6.1,  0.17),
    ("2.B.2.1", "2.B", 12.0, 0.14), ("2.B.2.2", "2.B", 12.0, 0.17), ("2.B.2.3", "2.B", 12.0, 0.21),
    ("3.A.1",   "3.A", 12.9, 0.11), ("3.A.2",   "3.A", 12.9, 0.18),
    ("3.B.1",   "3.B", 13.0, 0.11), ("3.B.2",   "3.B", 13.0, 0.18))

function regular_case_constructor(name, family, k, steepness)
    f = REGULAR_WAVE_FAMILIES[family]
    return FT -> regular_wave_case(FT; name, family, f.h, f.U₀, f.u_rms, f.w_rms, f.e, f.L, f.L_assumed, k, steepness)
end

const REGULAR_WAVE_CASES = Tuple(first.(REGULAR_WAVE_TABLE))

is_regular(case) = haskey(case, :regular)

const ANTI_STOKES_CASES = Dict("1.A" => case_1A, "1A" => case_1A,
                               "1.B" => case_1B, "1B" => case_1B,
                               "1.C.1" => case_1C1, "1C1" => case_1C1,
                               "1.C.2" => case_1C2, "1C2" => case_1C2,
                               "1.D" => case_1D, "1D" => case_1D)

for (name, family, k, steepness) in REGULAR_WAVE_TABLE
    ANTI_STOKES_CASES[name] = regular_case_constructor(name, family, k, steepness)
    ANTI_STOKES_CASES[replace(name, "." => "")] = ANTI_STOKES_CASES[name]
end

const WAVE_GROUP_CASES = ("1.A", "1.B", "1.C.1", "1.C.2", "1.D")

"""
    anti_stokes_case(name, FT=Float32)

Look up a case by name (for example `"1.D"`). Cases 2 and 3 (regular waves) are part of
the later physics campaign and are added here once transcribed.
"""
function anti_stokes_case(name, FT=Float32)
    haskey(ANTI_STOKES_CASES, name) ||
        error("Unknown case \"$name\". Known cases: $(join(sort(collect(keys(ANTI_STOKES_CASES))), ", "))")
    return ANTI_STOKES_CASES[name](FT)
end
