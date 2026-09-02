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

const ANTI_STOKES_CASES = Dict("1.A" => case_1A, "1A" => case_1A,
                               "1.B" => case_1B, "1B" => case_1B,
                               "1.C.1" => case_1C1, "1C1" => case_1C1,
                               "1.C.2" => case_1C2, "1C2" => case_1C2,
                               "1.D" => case_1D, "1D" => case_1D)

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
